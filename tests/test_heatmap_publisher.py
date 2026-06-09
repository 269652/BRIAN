# -*- coding: utf-8 -*-
"""TDD: HeatmapPublisher (L2b) — auto save + commit/push every N steps.

During long vast.ai / Colab runs the live heatmap must be persisted and
pushed to the repo on a configurable cadence so progress survives the
instance and is visible from anywhere. Git calls go through an injectable
runner so cadence + command construction are testable without a remote,
and any git failure is swallowed (never crashes training).
"""
import tempfile
from pathlib import Path

import pytest

from neuroslm.evolution.heatmap import TrainingHeatmap
from neuroslm.evolution.publisher import HeatmapPublisher


class _FakeRunner:
    """Records git invocations; configurable to fail."""
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, args, cwd=None):
        self.calls.append(list(args))
        if self.fail:
            raise RuntimeError("git boom")
        return 0


def _hm():
    hm = TrainingHeatmap()
    hm.update({"population:gws": 0.5}, kinds={"population:gws": "node"})
    return hm


class TestCadence:
    def test_publishes_only_on_multiples_of_n(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=500, runner=runner)
            assert pub.maybe_publish(_hm(), step=499) is False
            assert pub.maybe_publish(_hm(), step=500) is True
            assert pub.maybe_publish(_hm(), step=1000) is True
            assert pub.maybe_publish(_hm(), step=1001) is False

    def test_commit_every_zero_disables(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=0, runner=runner)
            assert pub.maybe_publish(_hm(), step=500) is False
            assert runner.calls == []


class TestPublishActions:
    def test_publish_saves_heatmap_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            pub = HeatmapPublisher(str(path), commit_every=1, runner=_FakeRunner())
            pub.maybe_publish(_hm(), step=1)
            assert path.exists()
            loaded = TrainingHeatmap.load(str(path))
            assert loaded.heat("population:gws") == pytest.approx(0.5)

    def test_publish_issues_add_commit_push(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=1, push=True,
                                   remote="origin", runner=runner)
            pub.maybe_publish(_hm(), step=1)
            verbs = [c[0] for c in runner.calls]
            assert verbs == ["add", "commit", "push"]
            # add references the heatmap path
            assert any(str(path) in arg for arg in runner.calls[0])
            # push targets the remote
            assert "origin" in runner.calls[2]

    def test_commit_message_includes_step(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=1, runner=runner)
            pub.maybe_publish(_hm(), step=7000)
            commit_call = next(c for c in runner.calls if c and c[0] == "commit")
            assert any("7000" in str(a) for a in commit_call)

    def test_push_false_skips_push(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=1, push=False,
                                   runner=runner)
            pub.maybe_publish(_hm(), step=1)
            verbs = [c[0] for c in runner.calls]
            assert "push" not in verbs
            assert verbs == ["add", "commit"]

    def test_explicit_branch_is_pushed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            runner = _FakeRunner()
            pub = HeatmapPublisher(str(path), commit_every=1, push=True,
                                   remote="origin", branch="heatmaps",
                                   runner=runner)
            pub.maybe_publish(_hm(), step=1)
            push_call = next(c for c in runner.calls if c and c[0] == "push")
            assert "origin" in push_call and "heatmaps" in push_call


class TestRobustness:
    def test_git_failure_is_swallowed(self):
        """A failing git call must not crash training; file still saved."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            pub = HeatmapPublisher(str(path), commit_every=1,
                                   runner=_FakeRunner(fail=True))
            # Should not raise.
            result = pub.maybe_publish(_hm(), step=1)
            assert result is True       # publish attempted
            assert path.exists()        # save happened before git
