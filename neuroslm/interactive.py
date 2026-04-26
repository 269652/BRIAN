"""Interactive shell.

Runs the brain continuously in a background thread. The user can type at
any time; their text is routed through the TextInputCortex (the
*external-text* sensory boundary), which:
  - injects the token IDs into the next cognitive context,
  - triggers an NE surge so the LC sharpens attention on new input,
  - decays in salience over subsequent ticks if no fresh input arrives.

The brain ticks autonomously even when the user is silent — those
"idle" ticks are mind-wandering: floating thought drifts, hippocampal
recall fires, motor head can still SPEAK if it wants to.

Display: brain emissions are word-buffered and printed on their own
`Brain>` lines; the user's input prompt line is cleared and re-drawn
around them so the two channels don't visually collide.

Usage:
  python -m neuroslm.interactive --ckpt checkpoints\\neuroslm_small_750.pt

Special commands:
  /quit            exit
  /state           dump current brain state
  /silence         disable model speaking until /unmute
  /unmute          re-enable speaking
  /clear           clear input cortex buffer & reset floating thought
  /tps N           set target ticks-per-second
  /temp X          set sampling temperature
  /topk N          set top-k
  /status          enable periodic status lines
  /nostatus        disable
"""
from __future__ import annotations
import argparse
import sys
import threading
import time
import queue
import torch

from .config import small, tiny, medium
from .brain import Brain
from .tokenizer import Tokenizer
from .modules.text_input import TextInputCortex
from .modules.motor import ACTION_NAMES, ACTION_INDEX
from .neurochem.transmitters import NT_NAMES


# ----------------------------------------------------------------------
# ANSI helpers (works in modern Windows Terminal / VS Code terminal)
# ----------------------------------------------------------------------
RESET = "\x1b[0m"
DIM   = "\x1b[2m"
BOLD  = "\x1b[1m"
CYAN  = "\x1b[36m"
GREEN = "\x1b[32m"
YEL   = "\x1b[33m"
RED   = "\x1b[31m"
MAG   = "\x1b[35m"
BLUE  = "\x1b[34m"

CLR_LINE = "\r\x1b[2K"   # carriage return + erase entire line


# ----------------------------------------------------------------------
# Display: a single lock guards stdout so the brain thread and the
# input thread never interleave half-written lines. The prompt is
# re-drawn after every brain message.
# ----------------------------------------------------------------------
_io_lock = threading.Lock()
_PROMPT = f"{BOLD}You>{RESET} "


def _safe_write(s: str):
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))
        sys.stdout.flush()


def print_above_prompt(s: str):
    """Print `s` (with trailing newline) above the current input prompt,
    then redraw the prompt so the user can keep typing."""
    with _io_lock:
        _safe_write(CLR_LINE + s + "\n" + _PROMPT)


def print_prompt():
    with _io_lock:
        _safe_write(_PROMPT)


# ----------------------------------------------------------------------
# Background brain runner
# ----------------------------------------------------------------------
class BrainRunner(threading.Thread):
    def __init__(self, brain: Brain, txt_in: TextInputCortex,
                 user_q: queue.Queue, ctrl: dict):
        super().__init__(daemon=True)
        self.brain = brain
        self.txt_in = txt_in
        self.user_q = user_q
        self.ctrl = ctrl
        self.stop_event = threading.Event()
        self.tick_count = 0
        self.silent_streak = 0
        # Boost period after fresh input: forces stronger SPEAK pressure
        # for a few ticks so the model attempts a response.
        self.boost_ticks_remaining = 0
        # Word-buffering: tokens are accumulated until we hit whitespace
        # or punctuation, then flushed as a chunk above the prompt.
        self.word_buf = ""
        self.tokens_since_flush = 0
        self.state = brain.init_latents(1, torch.device(ctrl["device"]))

    # ------------------------------------------------------------------
    def _flush_word_buf(self, force: bool = False):
        WORD_BREAKS = (" ", "\n", "\t", ".", "!", "?", ",", ";", ":", '"', "'")
        if not self.word_buf:
            return
        if force:
            chunk = self.word_buf.strip()
            if chunk:
                print_above_prompt(f"{GREEN}Brain>{RESET} {chunk}")
            self.word_buf = ""
            self.tokens_since_flush = 0
            return
        last_break = max((self.word_buf.rfind(c) for c in WORD_BREAKS),
                         default=-1)
        if last_break < 0:
            if self.tokens_since_flush > 24:
                self._flush_word_buf(force=True)
            return
        chunk = self.word_buf[: last_break + 1]
        self.word_buf = self.word_buf[last_break + 1:]
        chunk_strip = chunk.strip()
        if chunk_strip:
            print_above_prompt(f"{GREEN}Brain>{RESET} {chunk_strip}")
        self.tokens_since_flush = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _do_tick(self):
        cfg = self.brain.cfg
        device = torch.device(self.ctrl["device"])

        # 1) Drain pending user input into the text-input cortex.
        had_new_input = False
        while True:
            try:
                msg = self.user_q.get_nowait()
            except queue.Empty:
                break
            if self.ctrl.get("scaffold"):
                # Soft conversational frame — orients a TinyStories-trained
                # model toward producing a reply rather than continuing
                # whatever monologue it was generating.
                msg = f"\n\nUser said: {msg}\nThe story continues:"
            self.txt_in.receive(msg)
            had_new_input = True

        if had_new_input:
            self._flush_word_buf(force=True)
            # Reset floating thought so the brain pivots toward new input.
            self.state["floating_thought"].zero_()
            # Force speech for several ticks (motor SPEAK aux head is
            # still green at step 750 — give it explicit pressure).
            self.boost_ticks_remaining = self.ctrl["boost_ticks"]
            self.silent_streak = 0

        # 2) NE surge if salience just spiked (new input).
        ti_info = self.txt_in.step()
        if ti_info["ne_surge"] > 0:
            surge = torch.tensor([ti_info["ne_surge"]], device=device)
            self.brain.transmitters.release("NE", surge)

        # 3) Build context window from the text-input cortex's buffer.
        ids = self.txt_in.context_ids(cfg.lang_ctx, device)

        # 4) Run one cognitive tick.
        logits, self.state, info = self.brain.cognitive_step(ids, self.state)
        act = int(info["action_idx"][0].item())
        info["action_name"] = ACTION_NAMES[act]
        info["text_input"] = ti_info

        # 5) Decide: speak or stay silent?
        muted = self.ctrl.get("muted", False)
        in_boost = self.boost_ticks_remaining > 0
        if in_boost:
            self.boost_ticks_remaining -= 1
        force_speak = (in_boost
                       or self.silent_streak >= self.ctrl["max_silent_streak"])
        do_emit = ((act == ACTION_INDEX["SPEAK"]) or force_speak) and not muted

        if do_emit:
            t = self.ctrl["temperature"]
            k = self.ctrl["top_k"]
            next_logits = logits[:, -1] / max(t, 1e-5)
            if k:
                v, _ = next_logits.topk(k)
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tok_id = int(nxt[0, 0].item())
            self.txt_in.emit(tok_id)
            txt = self.txt_in.tokenizer.decode([tok_id])
            self.word_buf += txt
            self.tokens_since_flush += 1
            self._flush_word_buf(force=False)
            self.silent_streak = 0
        else:
            self.silent_streak += 1
            if self.silent_streak >= 6 and self.word_buf.strip():
                self._flush_word_buf(force=True)

        # 6) Status line every N ticks.
        self.tick_count += 1
        if self.ctrl.get("show_status") and \
           self.tick_count % self.ctrl["status_every"] == 0:
            self._print_status(info)

    def _print_status(self, info: dict):
        nt = info["nt"]
        ti = info["text_input"]
        line = (
            f"{DIM}[tick {self.tick_count:5d}] "
            f"act={info['action_name']:<14} "
            f"threat={float(info['threat']):.2f} "
            f"sal={ti['salience']:.2f} "
            f"buf={ti['buffer_len']:>4} "
            f"silent={self.silent_streak} "
            f"boost={self.boost_ticks_remaining} | "
            f"DA={nt['DA']:.2f} NE={nt['NE']:.2f} "
            f"5HT={nt['5HT']:.2f} ACh={nt['ACh']:.2f}{RESET}"
        )
        print_above_prompt(line)

    def run(self):
        target_dt = 1.0 / max(self.ctrl["tps"], 0.1)
        while not self.stop_event.is_set():
            t0 = time.time()
            try:
                self._do_tick()
            except Exception as e:  # noqa: BLE001
                print_above_prompt(
                    f"{RED}[brain error] {type(e).__name__}: {e}{RESET}")
                time.sleep(0.5)
            dt = time.time() - t0
            sleep = target_dt - dt
            if sleep > 0:
                time.sleep(sleep)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="small",
                    choices=["tiny", "small", "medium"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tps", type=float, default=4.0,
                    help="target cognitive ticks per second")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_silent_streak", type=int, default=6)
    ap.add_argument("--boost_ticks", type=int, default=24,
                    help="force SPEAK for this many ticks after fresh input")
    ap.add_argument("--scaffold", action="store_true",
                    help="wrap user input in a soft 'User said: ... story "
                         "continues:' frame so a TinyStories-trained model "
                         "orients toward replying")
    ap.add_argument("--status_every", type=int, default=20)
    ap.add_argument("--no_status", action="store_true")
    args = ap.parse_args()

    # Force UTF-8 + enable ANSI on Windows console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    presets = {"tiny": tiny, "small": small, "medium": medium}
    cfg = presets[args.preset]()
    tok = Tokenizer()
    cfg.vocab_size = tok.vocab_size
    brain = Brain(cfg)
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    brain.load_partial(sd)
    brain.to(args.device).eval()

    txt_in = TextInputCortex(tokenizer=tok)
    user_q: queue.Queue = queue.Queue()
    ctrl = {
        "device": args.device,
        "tps": args.tps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_silent_streak": args.max_silent_streak,
        "boost_ticks": args.boost_ticks,
        "scaffold": args.scaffold,
        "status_every": args.status_every,
        "show_status": not args.no_status,
        "muted": False,
    }

    runner = BrainRunner(brain, txt_in, user_q, ctrl)
    runner.start()

    print_above_prompt(
        f"{BOLD}{CYAN}NeuroSLM interactive shell{RESET}\n"
        f"{DIM}params: {brain.num_parameters()/1e6:.2f}M | "
        f"preset: {args.preset} | tps: {args.tps} | "
        f"temp: {args.temperature} | top_k: {args.top_k} | "
        f"scaffold: {args.scaffold}{RESET}\n"
        f"{DIM}Commands: /quit /state /silence /unmute /clear "
        f"/tps N /temp X /topk N /status /nostatus{RESET}"
    )

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                print_prompt()
                continue
            if line.startswith("/"):
                handle_command(line, ctrl, runner, brain)
                print_prompt()
            else:
                user_q.put(line)
                print_above_prompt(
                    f"{DIM}[queued — brain will attend in "
                    f"{ctrl['boost_ticks']} boosted ticks]{RESET}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        with _io_lock:
            _safe_write("\n" + DIM + "stopping brain...\n" + RESET)
        runner.stop_event.set()
        runner.join(timeout=2)


def handle_command(line: str, ctrl: dict, runner: BrainRunner, brain: Brain):
    parts = line.split()
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    def msg(s, color=YEL):
        print_above_prompt(f"{color}{s}{RESET}")

    if cmd in ("/quit", "/q", "/exit"):
        raise EOFError
    elif cmd == "/state":
        info = {
            "tick": runner.tick_count,
            "silent_streak": runner.silent_streak,
            "boost_remaining": runner.boost_ticks_remaining,
            "buf_len": len(runner.txt_in._buf),
            "salience": round(runner.txt_in.salience, 3),
            "muted": ctrl["muted"],
            "tps": ctrl["tps"],
            "temp": ctrl["temperature"],
            "top_k": ctrl["top_k"],
            "scaffold": ctrl["scaffold"],
            "NT": {n: round(float(brain.transmitters.get(n).mean()), 3)
                   for n in NT_NAMES},
        }
        msg(str(info), CYAN)
    elif cmd == "/silence":
        ctrl["muted"] = True
        msg("brain muted")
    elif cmd == "/unmute":
        ctrl["muted"] = False
        msg("brain unmuted")
    elif cmd == "/clear":
        runner.txt_in._buf.clear()
        runner.txt_in.salience = 0.0
        runner.state["floating_thought"].zero_()
        runner.word_buf = ""
        msg("input buffer + floating thought cleared")
    elif cmd == "/tps" and arg:
        ctrl["tps"] = float(arg)
        msg(f"tps -> {ctrl['tps']}")
    elif cmd == "/temp" and arg:
        ctrl["temperature"] = float(arg)
        msg(f"temperature -> {ctrl['temperature']}")
    elif cmd == "/topk" and arg:
        ctrl["top_k"] = int(arg)
        msg(f"top_k -> {ctrl['top_k']}")
    elif cmd == "/status":
        ctrl["show_status"] = True
        msg("status enabled")
    elif cmd == "/nostatus":
        ctrl["show_status"] = False
        msg("status disabled")
    else:
        msg(f"unknown command: {cmd}", RED)


if __name__ == "__main__":
    main()
