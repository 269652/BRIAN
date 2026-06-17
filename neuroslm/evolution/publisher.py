# -*- coding: utf-8 -*-
"""HeatmapPublisher — persist + commit/push the live heatmap on a cadence.

Long vast.ai / Colab runs update the heatmap continuously; this publisher
saves it and (optionally) commits and pushes it to the repo every N steps
so progress is durable and visible off-instance. Git calls go through an
injectable runner and every git failure is swallowed — publishing must
never crash training.

Auth for `git push` is whatever the run environment provides (e.g. a
tokenized remote URL on Colab, or a configured credential helper on vast).
This unit only issues the commands.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Callable, List, Optional


def _default_runner(args: List[str], cwd: Optional[str] = None) -> int:
    """Run `git <args>`; return the exit code, never raise."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return proc.returncode
    except Exception:
        return 1


class HeatmapPublisher:
    """Save the heatmap and commit/push it every ``commit_every`` steps."""

    def __init__(self, heatmap_path: str, commit_every: int = 500,
                 push: bool = True, remote: str = "origin",
                 branch: Optional[str] = None,
                 runner: Optional[Callable[..., int]] = None,
                 repo_root: Optional[str] = None,
                 dot_renderer: Optional[Callable] = None,
                 dot_path: Optional[str] = None,
                 png_renderer: Optional[Callable] = None,
                 png_path: Optional[str] = None) -> None:
        self.heatmap_path = heatmap_path
        self.commit_every = commit_every
        self.push = push
        self.remote = remote
        self.branch = branch
        self._run = runner or _default_runner
        self.repo_root = repo_root
        self.dot_renderer = dot_renderer  # callable(heatmap) -> dot str
        self.dot_path = dot_path          # path to write the .dot file
        self.png_renderer = png_renderer  # callable(heatmap, path) -> None; writes PNG
        self.png_path = png_path          # path for the rendered PNG

    def maybe_publish(self, heatmap, step: int) -> bool:
        """Publish iff ``commit_every > 0`` and ``step`` is a multiple of it."""
        if self.commit_every <= 0:
            return False
        if step % self.commit_every != 0:
            return False
        return self.publish(heatmap, step)

    def publish(self, heatmap, step: int) -> bool:
        """Save the heatmap (+ colored DOT when configured), then commit."""
        heatmap.save(self.heatmap_path)
        add_paths = [self.heatmap_path]

        if self.dot_renderer is not None and self.dot_path is not None:
            try:
                dot_src = self.dot_renderer(heatmap)
                Path(self.dot_path).write_text(dot_src, encoding="utf-8")
                add_paths.append(self.dot_path)
            except Exception:
                pass  # never crash — rendering failure is non-fatal

        if self.png_renderer is not None and self.png_path is not None:
            try:
                self.png_renderer(heatmap, self.png_path)
                add_paths.append(self.png_path)
            except Exception:
                pass  # never crash — rendering failure is non-fatal

        for path in add_paths:
            self._git(["add", path])
        self._git([
            "commit", "-m",
            f"chore(heatmap): update at step {step} [skip ci]",
        ])
        if self.push:
            target = self.branch or "HEAD"
            self._git(["push", self.remote, target])
        return True

    def _git(self, args: List[str]) -> None:
        try:
            self._run(args, cwd=self.repo_root)
        except Exception:
            pass  # never let a git error crash training
