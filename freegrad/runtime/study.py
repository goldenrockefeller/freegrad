"""Study specification and run expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from freegrad.runtime.git import GitProvenance
from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.run import RunSpec
from freegrad.runtime.study_manifest import StudyManifest


@dataclass(frozen=True)
class StudyExecutionPolicy:
    require_clean_git_for_new_study: bool = True
    require_confirmation_for_missing_git_provenance: bool = True
    require_confirmation_for_git_mismatch: bool = True


@dataclass(frozen=True)
class StudyPreflightWarning:
    message: str

#TODO define plotting classes for study/condition/runs. example if number of function_evals_elapsed to training curve plot.
@dataclass(frozen=True)
class StudySpec:
    name: str
    conditions: list[ConditionSpec]
    # TODO Remove this
    execution_policy: StudyExecutionPolicy = field(default_factory=StudyExecutionPolicy)

    def expand_runs(self, root_output_dir: Path) -> list[RunSpec]:
        runs: list[RunSpec] = []
        for condition in self.conditions:
            runs.extend(condition.expand_runs(root_output_dir, study_name=self.name))
        return runs

    # Remove or hard code this
    def preflight_warning(
        self,
        *,
        existing_manifest: StudyManifest | None,
        current_git: GitProvenance,
    ) -> StudyPreflightWarning | None:
        if existing_manifest is None:
            if self.execution_policy.require_clean_git_for_new_study and current_git.is_dirty:
                return StudyPreflightWarning(
                    message=(
                        f"Study '{self.name}' is starting on a dirty git branch.\n"
                        f"Current branch: {current_git.branch}\n"
                        f"Current commit: {current_git.commit_sha}"
                    )
                )
            return None

        if existing_manifest.git_provenance is None:
            if not self.execution_policy.require_confirmation_for_missing_git_provenance:
                return None
            return StudyPreflightWarning(
                message=(
                    f"Study '{self.name}' has existing outputs but no recorded git provenance.\n"
                    f"Current branch: {current_git.branch}\n"
                    f"Current commit: {current_git.commit_sha}\n"
                    "Continuing will record the current git state as the study baseline."
                )
            )

        differences = _git_differences(existing_manifest.git_provenance, current_git)
        if not differences or not self.execution_policy.require_confirmation_for_git_mismatch:
            return None

        dirty_suffix = "\nThe current working tree is also dirty." if current_git.is_dirty else ""
        difference_text = "\n".join(f"- {difference}" for difference in differences)
        return StudyPreflightWarning(
            message=(
                f"Study '{self.name}' is being resumed under different git provenance:\n"
                f"{difference_text}\n"
                f"Current branch: {current_git.branch}\n"
                f"Current commit: {current_git.commit_sha}"
                f"{dirty_suffix}"
            )
        )


def _git_differences(existing_git: GitProvenance, current_git: GitProvenance) -> list[str]:
    differences: list[str] = []
    if existing_git.branch != current_git.branch:
        differences.append(f"branch changed from '{existing_git.branch}' to '{current_git.branch}'")
    if existing_git.commit_sha != current_git.commit_sha:
        differences.append(f"commit changed from '{existing_git.commit_sha}' to '{current_git.commit_sha}'")
    return differences