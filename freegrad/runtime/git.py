"""Git provenance helpers for study execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GitProvenanceError(RuntimeError):
    """Raised when git provenance cannot be determined."""


class StudyExecutionAborted(RuntimeError):
    """Raised when study execution is aborted by a preflight guard."""


@dataclass(frozen=True)
class GitProvenance:
    branch: str
    commit_sha: str
    is_dirty: bool
    repository_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "is_dirty": self.is_dirty,
            "repository_root": self.repository_root,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> GitProvenance | None:
        if not payload:
            return None
        branch = payload.get("branch")
        commit_sha = payload.get("commit_sha")
        is_dirty = payload.get("is_dirty")
        if not isinstance(branch, str) or not isinstance(commit_sha, str) or not isinstance(is_dirty, bool):
            return None
        repository_root = payload.get("repository_root")
        if repository_root is not None and not isinstance(repository_root, str):
            repository_root = None
        return cls(
            branch=branch,
            commit_sha=commit_sha,
            is_dirty=is_dirty,
            repository_root=repository_root,
        )


def capture_git_provenance(cwd: Path | None = None) -> GitProvenance:
    working_dir = Path.cwd() if cwd is None else cwd
    try:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=working_dir)
        commit_sha = _run_git(["rev-parse", "HEAD"], cwd=working_dir)
        repository_root = _run_git(["rev-parse", "--show-toplevel"], cwd=working_dir)
        is_dirty = bool(_run_git(["status", "--porcelain"], cwd=working_dir))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GitProvenanceError("Unable to determine git provenance for this study.") from exc

    return GitProvenance(
        branch=branch,
        commit_sha=commit_sha,
        is_dirty=is_dirty,
        repository_root=repository_root,
    )


def _run_git(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()