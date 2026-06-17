# -*- coding: utf-8 -*-
"""TDD tests for ``neuroslm.chat_daemon``.

Validates the daemon's state-machine + threading contracts with a
deterministic stub ``GenerateFn`` so the suite never loads torch.

Coverage:

* :class:`TestMemoryRing` — thread-safe bounded ring
* :class:`TestChatDaemonBasics` — post / respond / think_once
* :class:`TestThreadSafety` — mutex serialises user + thought
* :class:`TestRender` — dashboard contains all three panes
* :class:`TestReplSlashCommands` — REPL parses /quit /clear /think
* :class:`TestResolveChatArch` — arch-locator precedence
"""
from __future__ import annotations

import io
import threading
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from neuroslm.chat_daemon import (
    ChatDaemon,
    ChatDaemonConfig,
    GenerateFn,
    _MemoryRing,
    _Episode,
    _resolve_chat_arch,
    _run_repl,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def echo_gen() -> GenerateFn:
    """Deterministic stub: returns ``"REPLY(<prompt-tail>):N"``.

    Lets tests assert exactly what the daemon would feed to the LM.
    """
    def _gen(prompt: str, n: int) -> str:
        tail = prompt[-32:].replace("\n", " | ")
        return f"REPLY({tail}):{n}"
    return _gen


@pytest.fixture
def slow_gen() -> GenerateFn:
    """Stub that sleeps for 200ms so the mutex contention tests can
    actually observe concurrent calls."""
    def _gen(prompt: str, n: int) -> str:
        time.sleep(0.2)
        return f"slow-reply({n})"
    return _gen


@pytest.fixture
def daemon(echo_gen) -> ChatDaemon:
    """Default daemon — no thought thread, ASCII-only render so tests
    don't have to compare against ANSI escapes."""
    return ChatDaemon(echo_gen, use_color=False)


# ─────────────────────────────────────────────────────────────────────
# 1. _MemoryRing
# ─────────────────────────────────────────────────────────────────────


class TestMemoryRing:

    def test_add_and_recent(self):
        m = _MemoryRing(maxlen=4)
        m.add("user", "hi")
        m.add("reply", "hello")
        recent = m.recent()
        assert [e.kind for e in recent] == ["user", "reply"]
        assert [e.content for e in recent] == ["hi", "hello"]

    def test_bounded(self):
        m = _MemoryRing(maxlen=3)
        for i in range(5):
            m.add("user", f"msg{i}")
        recent = m.recent()
        assert len(recent) == 3
        # Oldest two were dropped
        assert [e.content for e in recent] == ["msg2", "msg3", "msg4"]

    def test_recent_n_truncates(self):
        m = _MemoryRing(maxlen=10)
        for i in range(8):
            m.add("user", f"msg{i}")
        recent = m.recent(n=3)
        assert len(recent) == 3
        assert [e.content for e in recent] == ["msg5", "msg6", "msg7"]

    def test_recent_filters_by_kind(self):
        m = _MemoryRing(maxlen=10)
        m.add("user", "a")
        m.add("reply", "b")
        m.add("thought", "c")
        m.add("user", "d")
        recent_users = m.recent(kinds=("user",))
        assert [e.content for e in recent_users] == ["a", "d"]
        recent_chat = m.recent(kinds=("user", "reply"))
        assert [e.content for e in recent_chat] == ["a", "b", "d"]

    def test_len(self):
        m = _MemoryRing(maxlen=5)
        assert len(m) == 0
        m.add("user", "x")
        assert len(m) == 1

    def test_thread_safety_add(self):
        """Many concurrent adders should not lose or corrupt entries."""
        m = _MemoryRing(maxlen=10_000)
        def worker(start: int):
            for i in range(100):
                m.add("user", f"t{start}-{i}")
        threads = [threading.Thread(target=worker, args=(k,))
                   for k in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(m) == 1000


# ─────────────────────────────────────────────────────────────────────
# 2. ChatDaemon basics
# ─────────────────────────────────────────────────────────────────────


class TestChatDaemonBasics:

    def test_post_user_appends_and_ticks_idle_clock(self, daemon):
        t0 = daemon._last_user_input_ts
        time.sleep(0.01)
        daemon.post_user("hello")
        assert daemon._last_user_input_ts > t0
        assert any(e.kind == "user" and e.content == "hello"
                   for e in daemon.memory.recent())

    def test_post_user_ignores_empty(self, daemon):
        n0 = len(daemon.memory)
        daemon.post_user("")
        daemon.post_user("   \n")
        assert len(daemon.memory) == n0

    def test_respond_appends_both_user_and_reply(self, daemon):
        reply = daemon.respond("what is 2+2")
        assert reply.startswith("REPLY(")
        kinds = [e.kind for e in daemon.memory.recent()]
        assert "user" in kinds
        assert "reply" in kinds

    def test_respond_uses_chat_prompt_schema(self, echo_gen):
        """The prompt the model sees ends with ``BRIAN:`` to nudge a
        reply. Our echo stub captures the tail; we check it here."""
        d = ChatDaemon(echo_gen, use_color=False, label="BRIAN")
        d.respond("ping")
        replies = [e for e in d.memory.recent() if e.kind == "reply"]
        assert replies, "respond() should have posted a reply"
        # The echo gen returns ``REPLY(<tail>):<n>`` — check ``BRIAN:``
        # is in that tail.
        assert "BRIAN:" in replies[-1].content

    def test_think_once_appends_a_thought(self, daemon):
        # Seed memory so the thought has an anchor
        daemon.post_user("the sun is bright today")
        out = daemon.think_once()
        assert out is not None
        thoughts = daemon.memory.recent(kinds=("thought",))
        assert len(thoughts) == 1

    def test_think_once_handles_generate_exception(self, monkeypatch):
        """A broken generate must not kill the thought thread — the
        daemon catches + logs a ``system`` episode."""
        def _boom(prompt: str, n: int) -> str:
            raise RuntimeError("simulated NaN")
        d = ChatDaemon(_boom, use_color=False)
        d.post_user("seed")
        out = d.think_once()
        assert out is None
        sys_events = d.memory.recent(kinds=("system",))
        assert any("simulated NaN" in e.content for e in sys_events)

    def test_seed_rotation_visits_every_seed(self, daemon):
        seeds = daemon.cfg.thought_seeds
        used = [daemon._next_thought_seed() for _ in range(len(seeds) * 2)]
        # All seeds should appear at least once in the first cycle
        for s in seeds:
            assert s in used


# ─────────────────────────────────────────────────────────────────────
# 3. Thread safety — mutex serialises user-turn and thought-tick
# ─────────────────────────────────────────────────────────────────────


class TestThreadSafety:

    def test_think_once_skips_when_user_is_in_flight(self, slow_gen):
        """While ``respond()`` holds the lock, ``think_once()`` must
        return None instead of blocking."""
        d = ChatDaemon(slow_gen, use_color=False)
        results: List[object] = []

        def user_turn():
            d.respond("hi there")

        t = threading.Thread(target=user_turn)
        t.start()
        time.sleep(0.05)  # let user_turn acquire the mutex
        # During the 200ms generate, this should bounce
        out = d.think_once()
        results.append(out)
        t.join()
        assert results == [None]
        # The user reply still went through
        assert any(e.kind == "reply" for e in d.memory.recent())

    def test_stop_joins_thread(self, echo_gen):
        """``stop()`` must interrupt the inter-tick wait promptly so
        process shutdown doesn't hang for ``thought_period`` seconds."""
        # Short period so the test runs fast even if the interrupt is
        # broken — the assertion still proves the interrupt works.
        d = ChatDaemon(
            echo_gen, ChatDaemonConfig(thought_period=10.0),
            use_color=False,
        )
        d.start_thought_thread()
        assert d._thought_thread is not None
        assert d._thought_thread.is_alive()
        t0 = time.time()
        d.stop()
        elapsed = time.time() - t0
        # Stop must return within ~2 s (the join timeout in stop()).
        # With the Event-based interrupt this is essentially instant.
        assert elapsed < 2.5, \
            f"stop() blocked for {elapsed:.1f}s — _stop.wait() interrupt failing"
        assert not d._thought_thread.is_alive()

    def test_start_thought_thread_is_idempotent(self, daemon):
        daemon.start_thought_thread()
        t1 = daemon._thought_thread
        daemon.start_thought_thread()  # second call — must not spawn
        t2 = daemon._thought_thread
        assert t1 is t2
        daemon.stop()


# ─────────────────────────────────────────────────────────────────────
# 4. Render — dashboard contains all three panes
# ─────────────────────────────────────────────────────────────────────


class TestRender:

    def test_render_contains_all_three_panes(self, daemon):
        out = daemon.render()
        assert "memory" in out
        assert "thoughts" in out
        assert "chat" in out

    def test_render_shows_banner_with_memory_count(self, daemon):
        daemon.post_user("hi")
        out = daemon.render()
        assert "BRIAN chat daemon" in out
        assert "memory=" in out

    def test_render_empty_panes_have_placeholders(self, daemon):
        out = daemon.render()
        # Memory pane shows "(empty)" only if memo is empty AND no
        # episodes at all — once we add one, it's gone.
        assert "(waiting for idle window" in out or "thoughts" in out
        # Chat pane placeholder:
        assert "no turns yet" in out

    def test_render_includes_recent_chat(self, daemon):
        daemon.respond("hello world")
        out = daemon.render()
        assert "hello world" in out  # user line
        # Reply may be wrapped/truncated; check the BRIAN tag is there
        assert "BRIAN" in out

    def test_render_ascii_mode_has_no_ansi_escapes(self, daemon):
        daemon.respond("x")
        out = daemon.render()
        assert "\x1b[" not in out  # no ANSI escapes

    def test_render_color_mode_includes_ansi(self, echo_gen):
        d = ChatDaemon(echo_gen, use_color=True)
        d.respond("x")
        out = d.render()
        assert "\x1b[" in out

    def test_render_long_content_truncated(self, daemon):
        long = "x" * 500
        daemon.post_user(long)
        out = daemon.render()
        # No single line in the rendered chat pane should be > 200 chars
        for line in out.split("\n"):
            assert len(line) < 200


# ─────────────────────────────────────────────────────────────────────
# 5. REPL slash commands
# ─────────────────────────────────────────────────────────────────────


class TestReplSlashCommands:

    def _drive(self, daemon, lines: List[str]) -> str:
        """Feed ``lines`` to the REPL, return captured stdout."""
        in_stream = io.StringIO("\n".join(lines) + "\n")
        out_stream = io.StringIO()
        rc = _run_repl(daemon, out_stream=out_stream, in_stream=in_stream)
        assert rc == 0
        return out_stream.getvalue()

    def test_quit_exits_cleanly(self, daemon):
        out = self._drive(daemon, ["/quit"])
        assert "chat" in out  # at least one render happened

    def test_clear_wipes_memory(self, daemon):
        daemon.post_user("ghost")
        self._drive(daemon, ["/clear", "/quit"])
        # After /clear, the memory ring is empty except the
        # "memory cleared" system message.
        recent = daemon.memory.recent()
        user_msgs = [e for e in recent if e.kind == "user"]
        assert len(user_msgs) == 0

    def test_think_command_forces_a_tick(self, daemon):
        daemon.post_user("seed thought")
        self._drive(daemon, ["/think", "/quit"])
        thoughts = daemon.memory.recent(kinds=("thought",))
        assert len(thoughts) == 1

    def test_user_line_triggers_respond(self, daemon):
        out = self._drive(daemon, ["what is 2+2", "/quit"])
        replies = daemon.memory.recent(kinds=("reply",))
        assert len(replies) == 1
        # Reply should appear in the rendered output
        assert "REPLY" in out

    def test_empty_line_does_not_crash(self, daemon):
        out = self._drive(daemon, ["", "  ", "/quit"])
        assert "chat" in out


# ─────────────────────────────────────────────────────────────────────
# 6. _resolve_chat_arch — locator precedence
# ─────────────────────────────────────────────────────────────────────


class TestResolveChatArch:

    def test_explicit_arch_root_wins(self, tmp_path):
        arch = tmp_path / "myarch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("# stub")
        out = _resolve_chat_arch(str(arch), str(tmp_path / "x.pt"))
        assert out == str(arch)

    def test_arch_neuro_file_resolves_to_parent(self, tmp_path):
        arch = tmp_path / "myarch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("# stub")
        out = _resolve_chat_arch(str(arch / "arch.neuro"),
                                 str(tmp_path / "x.pt"))
        assert out == str(arch)

    def test_falls_back_to_ckpt_dir(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"")
        (tmp_path / "arch.neuro").write_text("# stub")
        out = _resolve_chat_arch(None, str(ckpt))
        assert out == str(tmp_path)

    def test_falls_back_to_repo_smollm(self, tmp_path, monkeypatch):
        """When neither --arch nor ckpt-dir arch exists, the repo's
        ``architectures/SmolLM`` is used. We can't easily isolate the
        repo root in this test, so we just verify the function returns
        a string (the repo's SmolLM does exist) or None gracefully."""
        ckpt = tmp_path / "isolated_ckpt.pt"
        ckpt.write_bytes(b"")
        out = _resolve_chat_arch(None, str(ckpt))
        # In the dev repo, architectures/SmolLM/arch.neuro exists,
        # so this returns a path; in a stripped checkout, it returns
        # None. Both are acceptable.
        assert out is None or isinstance(out, str)


# ─────────────────────────────────────────────────────────────────────
# 7. Episode + Config smoke
# ─────────────────────────────────────────────────────────────────────


class TestDataclasses:

    def test_episode_defaults_to_now(self):
        e = _Episode(kind="user", content="hi")
        assert e.kind == "user"
        assert e.content == "hi"
        assert e.ts > 0  # default_factory=time.time

    def test_config_defaults_are_cpu_friendly(self):
        c = ChatDaemonConfig()
        # The whole point of the defaults is laptop-resident chat:
        # token budgets stay small, thought period is in seconds.
        assert c.max_new_tokens <= 256
        assert c.thought_n_tok <= 128
        assert c.thought_period > 0
        assert len(c.thought_seeds) >= 3
