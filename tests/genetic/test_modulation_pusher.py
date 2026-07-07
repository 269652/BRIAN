# -*- coding: utf-8 -*-
"""Auto-push discovered modulations during a run (git add/commit/push the store)."""
import subprocess

from neuroslm.genetic.modulation_store import ModulationStore, ModulationRecord
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.modulation_pusher import push_modulations


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _init_repo(work, bare):
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True)
    work.mkdir()
    _git(["init"], work)
    _git(["checkout", "-b", "master"], work)
    _git(["config", "user.email", "t@t.t"], work)
    _git(["config", "user.name", "t"], work)
    _git(["remote", "add", "origin", str(bare)], work)
    (work / "README").write_text("x")
    _git(["add", "README"], work)
    _git(["commit", "-m", "init"], work)
    _git(["push", "origin", "master"], work)


def _gain():
    return Program([Instruction("tanh", "t2", ("t0",))], 4, 6, "t2")


class TestPush:
    def test_pushes_new_modulation_to_remote(self, tmp_path):
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        _init_repo(work, bare)
        store = ModulationStore(work / "modulations")
        store.save(ModulationRecord("gainA", _gain(), {"val_ppl": 7.1}))

        result = push_modulations(work, message="test push")
        assert result["pushed"] is True
        # the remote now has a commit touching modulations/gainA.neuro
        log = _git(["log", "--oneline", "--name-only", "master"], bare).stdout
        assert "modulations/gainA.neuro" in log

    def test_no_changes_is_a_clean_skip(self, tmp_path):
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        _init_repo(work, bare)
        (work / "modulations").mkdir()
        result = push_modulations(work, message="nothing")
        assert result["pushed"] is False
        assert "no changes" in result["reason"]

    def test_scopes_commit_to_modulations_only(self, tmp_path):
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        _init_repo(work, bare)
        # an unrelated dirty file must NOT be committed by the pusher
        (work / "other.txt").write_text("dirty")
        store = ModulationStore(work / "modulations")
        store.save(ModulationRecord("gainB", _gain(), {}))
        push_modulations(work, message="scoped")
        log = _git(["log", "--oneline", "--name-only", "master"], bare).stdout
        assert "modulations/gainB.neuro" in log
        assert "other.txt" not in log


class TestPushArtifacts:
    def test_pushes_multiple_artifact_paths(self, tmp_path):
        from neuroslm.genetic.modulation_pusher import push_artifacts
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        _init_repo(work, bare)
        (work / "modulations").mkdir()
        (work / "modulations" / "g.neuro").write_text("modulation g { program { t2 = tanh(t0)\nreturn t2 } }")
        (work / ".neuro").mkdir()
        (work / ".neuro" / "search_ledger.json").write_text("[]")
        res = push_artifacts(work, ["modulations", ".neuro/search_ledger.json"], message="arts")
        assert res["pushed"] is True
        log = _git(["log", "--oneline", "--name-only", "master"], bare).stdout
        assert "modulations/g.neuro" in log
        assert ".neuro/search_ledger.json" in log

    def test_absent_paths_are_a_clean_skip(self, tmp_path):
        from neuroslm.genetic.modulation_pusher import push_artifacts
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        _init_repo(work, bare)
        res = push_artifacts(work, ["nope", "also_nope"], message="x")
        assert res["pushed"] is False
        assert "no artifacts" in res["reason"]


class TestFreshRuntime:
    def test_commits_even_without_git_identity(self, tmp_path, monkeypatch):
        # a fresh Colab runtime has no user.name/user.email → `git commit` refuses.
        # push_artifacts must supply a fallback identity so the run still streams.
        import os
        from neuroslm.genetic.modulation_pusher import push_artifacts
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True)
        work.mkdir()
        _git(["init"], work)
        _git(["checkout", "-b", "master"], work)
        _git(["remote", "add", "origin", str(bare)], work)
        # deliberately NO user.email / user.name configured anywhere
        (work / "modulations").mkdir()
        (work / "modulations" / "g.neuro").write_text("modulation g { program { t2 = tanh(t0)\nreturn t2 } }")
        res = push_artifacts(work, ["modulations"], message="fresh runtime")
        assert res["pushed"] is True, res
        log = _git(["log", "--oneline", "--name-only", "master"], bare).stdout
        assert "modulations/g.neuro" in log


class TestConcurrentPush:
    def test_rebases_when_remote_moved(self, tmp_path):
        # simulate a concurrent writer: remote master advances between our commit
        # and push. The pusher must fetch+rebase and still land our artifact.
        from neuroslm.genetic.modulation_pusher import push_artifacts
        bare = tmp_path / "remote.git"
        work = tmp_path / "work"
        other = tmp_path / "other"
        _init_repo(work, bare)
        # a second clone pushes a new master commit (the "concurrent run")
        subprocess.run(["git", "clone", str(bare), str(other)], capture_output=True)
        _git(["checkout", "master"], other)
        _git(["config", "user.email", "o@o.o"], other)
        _git(["config", "user.name", "o"], other)
        (other / "concurrent.txt").write_text("from another run")
        _git(["add", "concurrent.txt"], other)
        _git(["commit", "-m", "concurrent log push"], other)
        _git(["push", "origin", "master"], other)

        # now our side commits an artifact against a STALE master and pushes
        (work / "modulations").mkdir()
        (work / "modulations" / "g.neuro").write_text("modulation g { program { t2 = tanh(t0)\nreturn t2 } }")
        res = push_artifacts(work, ["modulations"], message="ours")
        assert res["pushed"] is True   # rebased over the concurrent commit and landed
        log = _git(["log", "--oneline", "--name-only", "master"], bare).stdout
        assert "modulations/g.neuro" in log      # our artifact is there
        assert "concurrent.txt" in log           # and so is the concurrent commit
