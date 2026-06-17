"""Always-on chat daemon — boots BRIAN from a checkpoint and keeps the
model resident with three concurrent surfaces:

1. **Conversation surface** — user types, model answers, both append to
   episodic memory.

2. **Idle thoughts surface** — when the user is idle, the model
   self-prompts from its episodic memory + a small set of seed prompts.
   The thoughts append to a separate ring buffer that the dashboard
   shows live.

3. **CLI dashboard** — single-screen ANSI render with three panes:
   ``[memory]``, ``[thoughts]``, ``[chat]``. Refreshed on every input
   line and on every thought tick. Zero external deps (no ``rich``,
   no ``curses``) so this works on stock Windows + git-bash + Linux.

Architecture
------------
::

    ┌──────────────────────────────────────────────────────┐
    │ MainThread                                           │
    │ ─ reads stdin, runs generate(), updates state        │
    │   ─ posts user msg to EpisodicMemory                 │
    │   ─ calls model.generate() → reply                   │
    │   ─ posts reply to EpisodicMemory + chat ring        │
    │   ─ re-renders dashboard                             │
    │                                                      │
    │ ThoughtThread (daemon)                               │
    │ ─ wakes every ``thought_period`` seconds while idle  │
    │ ─ samples a seed from EpisodicMemory + thought-seeds │
    │ ─ runs a SHORT generate() (n_tokens ≤ thought_n_tok) │
    │ ─ posts to thoughts ring + EpisodicMemory            │
    │ ─ skips ticks while a user generate() is in flight   │
    │   (mutex on ``_inference_lock``)                     │
    └──────────────────────────────────────────────────────┘

The model + tokenizer are loaded ONCE at boot. Generation is
synchronous within a single thread (transformer KV-cache state is
not thread-safe), so the inference mutex serialises user-turn and
thought-tick calls. CPU runs are fully supported; the daemon is
useful for keeping a checkpoint warm on a laptop.

Public entrypoints
------------------
* :class:`ChatDaemon` — instance API for tests / programmatic use.
* :func:`run_chat_daemon(args)` — the ``brian chat`` CLI entrypoint.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, List, Optional, Tuple


# Import-safe constants. The module must NOT import torch / harness at
# top level; tests for the dashboard / threading should not need a
# torch install. The ChatDaemon constructor takes a generate callable
# so the torch path is fully injectable.


# ── ANSI helpers (zero-dep dashboard) ───────────────────────────────

ANSI_RESET = "\x1b[0m"
ANSI_DIM = "\x1b[2m"
ANSI_BOLD = "\x1b[1m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"
ANSI_GREEN = "\x1b[32m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_RED = "\x1b[31m"
ANSI_CLEAR = "\x1b[2J"
ANSI_HOME = "\x1b[H"


def _ansi(s: str, code: str) -> str:
    """Wrap ``s`` in ``code`` ... ``ANSI_RESET``. Used by the renderer
    so the test harness can monkey-patch out colour by setting
    ``ChatDaemon._use_color = False``.
    """
    return f"{code}{s}{ANSI_RESET}"


# ── Episodic memory shim ─────────────────────────────────────────────
# The project already has ``neuroslm.memory.episodic.EpisodicMemory`` —
# a thread-safe ring buffer. We use it but avoid a hard import so unit
# tests can run without ``threading`` lock contention.

@dataclass
class _Episode:
    """One memory entry: kind ∈ {'user', 'reply', 'thought', 'system'}."""
    kind: str
    content: str
    ts: float = field(default_factory=time.time)


class _MemoryRing:
    """Thread-safe bounded ring of :class:`_Episode`. Drop-in for the
    dashboard — the project's own ``EpisodicMemory`` is heavier
    (numeric vectors, NT state, emotion tags) and we don't need that
    surface here.
    """

    def __init__(self, maxlen: int = 256) -> None:
        self._buf: Deque[_Episode] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, kind: str, content: str) -> None:
        with self._lock:
            self._buf.append(_Episode(kind=kind, content=content))

    def recent(self, n: int = 32, kinds: Optional[Tuple[str, ...]] = None
               ) -> List[_Episode]:
        with self._lock:
            seq = list(self._buf)
        if kinds is None:
            return seq[-n:]
        return [e for e in seq if e.kind in kinds][-n:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ── Generate adapter ─────────────────────────────────────────────────
# A ``GenerateFn`` is anything that maps ``(prompt, max_new_tokens) → str``.
# This is the ONE seam between the daemon and a torch-backed harness;
# it lets the test suite inject a deterministic stub.

GenerateFn = Callable[[str, int], str]


def _build_generate_fn_from_harness(
        harness: Any,
        tokenizer: Any,
        device: str = "cpu",
        temperature: float = 0.8,
        top_k: int = 40,
) -> GenerateFn:
    """Wrap a BRIAN harness + tokenizer into a :data:`GenerateFn`.

    Lazy-imports torch so the daemon module stays import-clean.
    Greedy/top-k sampling on the LM head — no beam search; this is a
    chat daemon, not a benchmark runner. Caller controls
    ``max_new_tokens`` per call so user turns can be longer than
    background thoughts.

    Notes
    -----
    The harness is expected to expose ``language_model`` (the DSL
    LanguageCortex / N8 cortex). When a harness has no LM head (e.g.
    a pure circuit harness), this raises ``ValueError`` — the daemon
    requires LM-head-equipped checkpoints.
    """
    import torch
    import torch.nn.functional as F

    lm = getattr(harness, "language_model", None) or harness
    if lm is None:
        raise ValueError(
            "ChatDaemon: harness has no language_model attribute; "
            "this daemon requires an LM-head-equipped checkpoint "
            "(use the dsl_lm model, not circuit).")

    @torch.no_grad()
    def _generate(prompt: str, max_new_tokens: int) -> str:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            # Single-batch, naïve forward — the chat daemon optimises
            # for low-latency-feeling on CPU rather than throughput.
            logits = lm(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Some cortexes return ``(B, T, V)`` and some return the
            # final-token slice ``(B, V)``; handle both.
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                cutoff = vals[..., -1:].expand_as(logits)
                logits = torch.where(logits < cutoff,
                                     torch.full_like(logits, float("-inf")),
                                     logits)
            probs = F.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, num_samples=1).item())
            out_ids.append(tok)
            x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=-1)
            # Cheap stop heuristic: if the tokenizer round-trips a
            # double-newline, we're done with the turn.
            if len(out_ids) >= 8:
                tail = tokenizer.decode(out_ids[-8:])
                if "\n\n" in tail:
                    break
        return tokenizer.decode(out_ids)

    return _generate


# ── Daemon ───────────────────────────────────────────────────────────


@dataclass
class ChatDaemonConfig:
    """All knobs in one place. Defaults are tuned for CPU laptops."""

    max_new_tokens: int = 96
    """Token budget for a user-turn reply."""

    thought_n_tok: int = 32
    """Token budget for a single idle-thought."""

    thought_period: float = 12.0
    """Seconds between idle-thought ticks."""

    idle_threshold: float = 6.0
    """Seconds of user inactivity before thoughts start firing."""

    memory_size: int = 512
    """Episodic ring depth (kept small enough to print quickly)."""

    chat_visible: int = 12
    """How many recent chat turns the dashboard shows."""

    thoughts_visible: int = 6
    """How many recent thoughts the dashboard shows."""

    memory_visible: int = 6
    """How many recent memory entries (any kind) the dashboard shows."""

    thought_seeds: Tuple[str, ...] = (
        "I just thought about ",
        "What I've been wondering: ",
        "Continuing my last train of thought, ",
        "On reflection, ",
        "The pattern I see is ",
    )
    """Seeds prepended to the most-recent memory entry to nudge the
    model into a self-reflective continuation rather than a literal
    repeat."""


class ChatDaemon:
    """Boot a checkpointed BRIAN model and keep it talking.

    The daemon is a small state machine:

    * a memory ring (:class:`_MemoryRing`)
    * a thoughts thread (idle-tick → ``generate_fn``)
    * a render method (:meth:`render` → str)
    * a single mutex around ``generate_fn`` so user turns never race
      thought ticks

    Parameters
    ----------
    generate_fn : :data:`GenerateFn`
        ``(prompt, max_new_tokens) → str``. Tests pass a deterministic
        stub; production passes the harness-wrapped function from
        :func:`_build_generate_fn_from_harness`.
    cfg : :class:`ChatDaemonConfig`, optional
        All threading / token-budget knobs.
    label : str, default ``"BRIAN"``
        Shown in the dashboard banner.
    use_color : bool, default ``True``
        Disable to make the dashboard plain ASCII (for tests / logs).
    """

    def __init__(
            self,
            generate_fn: GenerateFn,
            cfg: Optional[ChatDaemonConfig] = None,
            *,
            label: str = "BRIAN",
            use_color: bool = True,
    ) -> None:
        self._gen = generate_fn
        self.cfg = cfg or ChatDaemonConfig()
        self.label = label
        self._use_color = use_color
        self.memory = _MemoryRing(maxlen=self.cfg.memory_size)
        self._inference_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_user_input_ts: float = time.time()
        self._thought_thread: Optional[threading.Thread] = None
        self._is_thinking: bool = False
        # Seed counter for round-robin thought-seed selection
        self._seed_idx: int = 0

    # ── State updates ───────────────────────────────────────────────

    def post_user(self, text: str) -> None:
        """Append a user turn to memory + tick the idle clock."""
        text = text.strip()
        if not text:
            return
        self.memory.add("user", text)
        self._last_user_input_ts = time.time()

    def post_reply(self, text: str) -> None:
        """Append the model's user-turn reply."""
        if text:
            self.memory.add("reply", text.strip())

    def post_thought(self, text: str) -> None:
        """Append an idle thought."""
        if text:
            self.memory.add("thought", text.strip())

    def post_system(self, text: str) -> None:
        """Append a system message (boot stamp, ckpt resume, etc)."""
        if text:
            self.memory.add("system", text.strip())

    # ── Inference seams ─────────────────────────────────────────────

    def respond(self, user_text: str) -> str:
        """Run the user-turn generate. Posts user + reply + returns
        the reply string."""
        self.post_user(user_text)
        prompt = self._build_chat_prompt()
        with self._inference_lock:
            reply = self._gen(prompt, self.cfg.max_new_tokens)
        self.post_reply(reply)
        return reply

    def think_once(self) -> Optional[str]:
        """Run one idle-thought tick. Returns the thought text, or
        ``None`` if the daemon is mid-conversation (mutex held).

        This is the unit-of-work the thought thread loops over. It
        never blocks waiting for the lock — a busy lock means the
        user is mid-turn and the thought tick simply skips.
        """
        if not self._inference_lock.acquire(blocking=False):
            return None
        try:
            self._is_thinking = True
            seed = self._next_thought_seed()
            recent = self.memory.recent(8)
            anchor = recent[-1].content if recent else ""
            prompt = (seed + anchor)[: 400]
            thought = self._gen(prompt, self.cfg.thought_n_tok)
            self.post_thought(thought)
            return thought
        except Exception as e:
            self.post_system(f"[thought-error] {type(e).__name__}: {e}")
            return None
        finally:
            self._is_thinking = False
            self._inference_lock.release()

    def _next_thought_seed(self) -> str:
        """Round-robin over ``cfg.thought_seeds``."""
        seeds = self.cfg.thought_seeds or ("",)
        s = seeds[self._seed_idx % len(seeds)]
        self._seed_idx += 1
        return s

    def _build_chat_prompt(self) -> str:
        """Concatenate the recent chat context into a single prompt.

        Uses an extremely simple ``USER:``/``BRIAN:`` schema. We rely
        on the model having seen enough chat-mix data during training
        to follow the format; it is *not* an instruction-tuned model
        so quality varies with the checkpoint.
        """
        turns = self.memory.recent(8, kinds=("user", "reply"))
        lines: List[str] = []
        for t in turns:
            tag = "USER" if t.kind == "user" else self.label.upper()
            lines.append(f"{tag}: {t.content}")
        lines.append(f"{self.label.upper()}:")
        return "\n".join(lines)

    # ── Thread management ───────────────────────────────────────────

    def start_thought_thread(self) -> None:
        """Spawn the idle-thought daemon thread (no-op if running)."""
        if self._thought_thread and self._thought_thread.is_alive():
            return
        t = threading.Thread(
            target=self._thought_loop, daemon=True, name="brian-thoughts")
        self._thought_thread = t
        t.start()

    def stop(self) -> None:
        """Signal the thought thread to exit; join with a short
        timeout so a hung generate() can't keep the process alive."""
        self._stop.set()
        if self._thought_thread and self._thought_thread.is_alive():
            self._thought_thread.join(timeout=2.0)

    def _thought_loop(self) -> None:
        """Body of the thought thread.

        Uses ``_stop.wait()`` instead of ``time.sleep()`` so that a
        ``stop()`` call interrupts the inter-tick wait immediately —
        otherwise process shutdown would hang for up to
        ``thought_period`` seconds (default 12s) and the join in
        ``stop()`` would time out leaving an orphan daemon thread.
        """
        while not self._stop.is_set():
            # wait() returns True if the event fires, False on timeout.
            if self._stop.wait(self.cfg.thought_period):
                break
            idle_for = time.time() - self._last_user_input_ts
            if idle_for < self.cfg.idle_threshold:
                continue
            self.think_once()

    # ── Rendering ───────────────────────────────────────────────────

    def render(self) -> str:
        """Compose the dashboard as one big string. Caller writes it
        to stdout (the daemon never owns stdout directly so tests
        can capture the rendered text).
        """
        col = self._use_color
        memo = self.memory.recent(self.cfg.memory_visible)
        thoughts = self.memory.recent(
            self.cfg.thoughts_visible * 2, kinds=("thought",)
        )[-self.cfg.thoughts_visible:]
        chat = self.memory.recent(
            self.cfg.chat_visible * 2, kinds=("user", "reply")
        )[-self.cfg.chat_visible:]

        def header(text: str, color: str) -> str:
            line = f"── {text} " + "─" * max(2, 60 - len(text))
            return _ansi(line, color) if col else line

        def episode_line(e: _Episode, max_len: int = 80) -> str:
            ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
            content = e.content.replace("\n", " ⏎ ")
            if len(content) > max_len:
                content = content[:max_len - 1] + "…"
            tag = {
                "user":    "USER  ",
                "reply":   "BRIAN ",
                "thought": "💭    ",
                "system":  "SYS   ",
            }.get(e.kind, "?     ")
            colour = {
                "user":    ANSI_CYAN,
                "reply":   ANSI_GREEN,
                "thought": ANSI_YELLOW,
                "system":  ANSI_DIM,
            }.get(e.kind, "")
            line = f"  [{ts}] {tag} {content}"
            return _ansi(line, colour) if col and colour else line

        out: List[str] = []
        banner = (f"{self.label} chat daemon — "
                  f"memory={len(self.memory)}/{self.cfg.memory_size} "
                  f"{'(thinking…)' if self._is_thinking else ''}")
        out.append(_ansi(banner, ANSI_BOLD) if col else banner)
        out.append("")
        out.append(header("memory", ANSI_MAGENTA))
        if not memo:
            out.append("  (empty)")
        for e in memo:
            out.append(episode_line(e))
        out.append("")
        out.append(header("thoughts", ANSI_YELLOW))
        if not thoughts:
            out.append("  (waiting for idle window …)")
        for e in thoughts:
            out.append(episode_line(e))
        out.append("")
        out.append(header("chat", ANSI_CYAN))
        if not chat:
            out.append("  (no turns yet — type below to talk to BRIAN)")
        for e in chat:
            out.append(episode_line(e))
        out.append("")
        return "\n".join(out)


# ── CLI entrypoint ───────────────────────────────────────────────────


def run_chat_daemon(
        ckpt_path: str,
        *,
        arch_root: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.8,
        top_k: int = 40,
        max_new_tokens: int = 96,
        thought_n_tok: int = 32,
        thought_period: float = 12.0,
        idle_threshold: float = 6.0,
        no_color: bool = False,
        no_thoughts: bool = False,
        out_stream=sys.stdout,
        in_stream=sys.stdin,
) -> int:
    """Boot a checkpoint and run the interactive dashboard loop.

    Wired into ``brian chat`` by :func:`neuroslm.cli.cmd_chat`.

    Returns 0 on clean exit (Ctrl-D, ``/quit``), 1 on boot failure.

    The boot sequence is:

    1. Locate ``arch.neuro`` (from ``arch_root`` arg, or the
       ``arch_path`` recorded in the checkpoint, or
       ``architectures/SmolLM`` as last resort).
    2. Build a DSL LM harness sized to the checkpoint.
    3. Call ``harness.load_checkpoint(ckpt_path)``.
    4. Wrap into a :data:`GenerateFn` and start the daemon.
    """
    # Resolve arch root
    arch_path = _resolve_chat_arch(arch_root, ckpt_path)
    if arch_path is None:
        print(
            "[chat] ✗ cannot find an arch.neuro to boot. "
            "Pass --arch <path>, or run from the repo root.",
            file=sys.stderr)
        return 1

    out_stream.write(f"[chat] booting from arch={arch_path}\n")
    out_stream.write(f"[chat] checkpoint={ckpt_path}\n")
    out_stream.flush()

    try:
        from neuroslm.train_dsl import build_dsl_lm_harness, _load_tokenizer
        from neuroslm.dsl.training_config import load_training_config_from_arch
        cfg = load_training_config_from_arch(Path(arch_path))
        tok = _load_tokenizer()
        # Pull dims from the active scale variant if any, else fall
        # back to the trainer defaults. Same precedence as
        # train_dsl.main().
        d_model = 256
        depth = 6
        n_heads = 8
        max_ctx = 1024
        try:
            scale = cfg.scales.variants.get(cfg.scales.default)
            if scale is not None:
                d_model = int(scale.d_model)
                depth = int(scale.depth)
                n_heads = int(scale.n_heads)
                max_ctx = int(getattr(scale, "seq_len", max_ctx))
        except Exception:
            pass
        harness = build_dsl_lm_harness(
            arch_root=Path(arch_path),
            vocab_size=tok.vocab_size,
            d_model=d_model, depth=depth, n_heads=n_heads,
            max_ctx=max_ctx, device=device,
        )
        step = harness.load_checkpoint(ckpt_path, device=device)
        out_stream.write(
            f"[chat] ✓ loaded step={step} d_model={d_model} "
            f"depth={depth} heads={n_heads}\n")
        out_stream.flush()
    except Exception as e:
        print(f"[chat] ✗ boot failure: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    gen_fn = _build_generate_fn_from_harness(
        harness, tok, device=device, temperature=temperature, top_k=top_k)
    daemon = ChatDaemon(
        gen_fn,
        ChatDaemonConfig(
            max_new_tokens=max_new_tokens,
            thought_n_tok=thought_n_tok,
            thought_period=thought_period,
            idle_threshold=idle_threshold,
        ),
        use_color=(not no_color) and out_stream.isatty(),
    )
    daemon.post_system(
        f"booted from {os.path.basename(ckpt_path)} @ step {step}"
    )
    if not no_thoughts:
        daemon.start_thought_thread()

    return _run_repl(daemon, out_stream=out_stream, in_stream=in_stream)


def _resolve_chat_arch(arch_root: Optional[str], ckpt_path: str
                       ) -> Optional[str]:
    """Locate the ``arch.neuro`` to boot. Precedence:

    1. ``--arch`` CLI arg.
    2. The directory containing the ``.pt`` (sometimes the ckpt is
       saved next to its arch).
    3. ``architectures/SmolLM`` under the repo root.
    """
    if arch_root:
        p = Path(arch_root)
        if (p / "arch.neuro").is_file():
            return str(p)
        if p.is_file() and p.parent.is_dir() \
                and (p.parent / "arch.neuro").is_file():
            return str(p.parent)
    ckpt_dir = Path(ckpt_path).parent
    if (ckpt_dir / "arch.neuro").is_file():
        return str(ckpt_dir)
    repo_root = Path(__file__).resolve().parent.parent
    fallback = repo_root / "architectures" / "SmolLM"
    if (fallback / "arch.neuro").is_file():
        return str(fallback)
    return None


def _run_repl(
        daemon: ChatDaemon,
        *,
        out_stream=sys.stdout,
        in_stream=sys.stdin,
) -> int:
    """Read-eval-print loop on top of ``daemon``.

    Built-in slash commands:

    * ``/quit`` — exit cleanly
    * ``/clear`` — clear the memory ring
    * ``/think`` — force a thought tick
    * ``/render`` — re-print the dashboard

    Anything else is forwarded as a user turn.
    """
    out_stream.write(daemon.render())
    out_stream.write("\n> ")
    out_stream.flush()
    try:
        while True:
            line = in_stream.readline()
            if not line:
                break  # EOF
            line = line.strip()
            if not line:
                out_stream.write("> ")
                out_stream.flush()
                continue
            if line in ("/quit", "/exit", ":q"):
                break
            if line == "/clear":
                # Best-effort wipe — we drop everything and re-banner.
                daemon.memory = _MemoryRing(
                    maxlen=daemon.cfg.memory_size)
                daemon.post_system("memory cleared")
            elif line == "/think":
                daemon.think_once()
            elif line == "/render":
                pass
            else:
                reply = daemon.respond(line)
                # Printed inside the chat pane on next render
                _ = reply
            out_stream.write(daemon.render())
            out_stream.write("\n> ")
            out_stream.flush()
    except KeyboardInterrupt:
        out_stream.write("\n[chat] ^C — exiting\n")
    finally:
        daemon.stop()
    return 0


__all__ = [
    "ChatDaemon",
    "ChatDaemonConfig",
    "GenerateFn",
    "run_chat_daemon",
]
