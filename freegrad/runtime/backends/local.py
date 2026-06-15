"""Sequential local runtime backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from freegrad.runtime.backends.base import Backend, StudyConfirmationHandler
from freegrad.runtime.condition import ConditionSpec
from freegrad.runtime.git import GitProvenance, StudyExecutionAborted, capture_git_provenance
from freegrad.runtime.run import RunResult, RunSpec, run_one
from freegrad.runtime.study import StudySpec
from freegrad.runtime.study_manifest import (
    is_completed_run,
    load_study_manifest,
    should_resume_run,
    skipped_run_result,
    write_study_manifest,
)


@dataclass
class LocalBackend(Backend):
    root_output_dir: Path = Path("results")
    confirmation_handler: StudyConfirmationHandler | None = None

    def run(self, run_spec: RunSpec) -> RunResult:
        return run_one(run_spec)

    def run_condition(self, condition: ConditionSpec) -> list[RunResult]:
        condition_root = self.root_output_dir / condition.name
        return [self.run(run_spec) for run_spec in condition.expand_runs(condition_root)]

    def run_study(self, study: StudySpec) -> list[RunResult]:
        study_root = self.root_output_dir / study.name
        run_specs = study.expand_runs(study_root)
        existing_manifest = load_study_manifest(study_root)
        current_git = self._current_git_provenance()
        self._preflight_study_execution(study=study, existing_manifest=existing_manifest, current_git=current_git)
        write_study_manifest(study.name, study_root, run_specs, git_provenance=current_git)

        results: list[RunResult] = []
        for run_spec in run_specs:
            if is_completed_run(run_spec):
                results.append(skipped_run_result(run_spec))
                continue

            effective_run_spec = run_spec
            if should_resume_run(run_spec):
                effective_run_spec = replace(
                    run_spec,
                    training_config={**run_spec.training_config, "resume": True},
                )

            results.append(self.run(effective_run_spec))
            write_study_manifest(study.name, study_root, run_specs, git_provenance=current_git)

        return results

    def _current_git_provenance(self) -> GitProvenance:
        return capture_git_provenance()

    def _preflight_study_execution(
        self,
        *,
        study: StudySpec,
        existing_manifest,
        current_git: GitProvenance,
    ) -> None:
        # TODO warning should be hard coded, not apart of study
        warning = study.preflight_warning(existing_manifest=existing_manifest, current_git=current_git)
        if warning is not None:
            self._confirm_or_abort(warning.message)

    # TODO warning should be hard coded
    def _confirm_or_abort(self, message: str) -> None:
        if self.confirmation_handler is None:
            raise StudyExecutionAborted(f"{message}\nConfirmation required, but no study confirmation handler is configured.")
        if not self.confirmation_handler.confirm_study_execution(message):
            raise StudyExecutionAborted("Study execution aborted by user.")
