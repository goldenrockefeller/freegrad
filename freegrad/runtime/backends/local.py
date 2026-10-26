"""Sequential local runtime backend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

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
    progress_callback: Callable[[str], None] | None = None

    def run(self, run_spec: RunSpec) -> RunResult:
        return run_one(run_spec, progress_callback=self.progress_callback)

    def run_condition(self, condition: ConditionSpec) -> list[RunResult]:
        condition_root = self.root_output_dir / condition.name
        run_specs = condition.expand_runs(condition_root)
        self._report(
            f"[condition] START name={condition.name} runs={len(run_specs)} "
            "mode=sequential concurrency=1"
        )
        results = [
            self._run_with_events(run_spec, index=index, total=len(run_specs))
            for index, run_spec in enumerate(run_specs, start=1)
        ]
        self._report_summary("condition", condition.name, results)
        return results

    def run_study(self, study: StudySpec) -> list[RunResult]:
        study_root = self.root_output_dir / study.name
        run_specs = study.expand_runs(study_root)
        self._report(
            f"[study] START name={study.name} conditions={len(study.conditions)} "
            f"runs={len(run_specs)} mode=sequential concurrency=1"
        )
        existing_manifest = load_study_manifest(study_root)
        current_git = self._current_git_provenance()
        self._preflight_study_execution(study=study, existing_manifest=existing_manifest, current_git=current_git)
        write_study_manifest(study.name, study_root, run_specs, git_provenance=current_git)

        condition_counts = Counter(run_spec.condition_name for run_spec in run_specs)
        condition_results: dict[str, list[RunResult]] = {}
        active_condition: str | None = None
        results: list[RunResult] = []
        for index, run_spec in enumerate(run_specs, start=1):
            if run_spec.condition_name != active_condition:
                if active_condition is not None:
                    self._report_summary(
                        "condition",
                        active_condition,
                        condition_results[active_condition],
                    )
                active_condition = run_spec.condition_name
                condition_results.setdefault(active_condition, [])
                self._report(
                    f"[condition] START name={active_condition} "
                    f"runs={condition_counts[active_condition]}"
                )

            if is_completed_run(run_spec):
                result = skipped_run_result(run_spec)
                results.append(result)
                condition_results[run_spec.condition_name].append(result)
                self._report(
                    f"[run {index}/{len(run_specs)}] SKIP name={run_spec.name} "
                    "reason=already_completed"
                )
                continue

            effective_run_spec = run_spec
            resuming = should_resume_run(run_spec)
            if resuming:
                effective_run_spec = replace(
                    run_spec,
                    training_config={**run_spec.training_config, "resume": True},
                )

            result = self._run_with_events(
                effective_run_spec,
                index=index,
                total=len(run_specs),
                resuming=resuming,
            )
            results.append(result)
            condition_results[run_spec.condition_name].append(result)
            write_study_manifest(study.name, study_root, run_specs, git_provenance=current_git)

        if active_condition is not None:
            self._report_summary("condition", active_condition, condition_results[active_condition])
        self._report_summary("study", study.name, results)
        return results

    def _run_with_events(
        self,
        run_spec: RunSpec,
        *,
        index: int,
        total: int,
        resuming: bool = False,
    ) -> RunResult:
        resume_text = " resume=true" if resuming else ""
        self._report(
            f"[run {index}/{total}] START name={run_spec.name} "
            f"condition={run_spec.condition_name} seed={run_spec.seed}{resume_text}"
        )
        started = perf_counter()
        result = self.run(run_spec)
        elapsed = perf_counter() - started
        details = _result_metric_summary(result)
        suffix = f" {details}" if details else ""
        self._report(
            f"[run {index}/{total}] {result.status} name={run_spec.name} "
            f"elapsed={elapsed:.1f}s{suffix} output={result.output_dir}"
        )
        return result

    def _report_summary(self, kind: str, name: str, results: list[RunResult]) -> None:
        completed = sum(result.status == "COMPLETED" for result in results)
        failed = sum(result.status != "COMPLETED" for result in results)
        status = "COMPLETED" if failed == 0 else "FAILED"
        self._report(
            f"[{kind}] {status} name={name} completed={completed} failed={failed}"
        )

    def _report(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

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


def _result_metric_summary(result: RunResult) -> str:
    details = []
    accuracy = result.final_metrics.get("eval/accuracy")
    loss = result.final_metrics.get("eval/loss")
    if isinstance(accuracy, (int, float)):
        details.append(f"eval_accuracy={accuracy:.4f}")
    if isinstance(loss, (int, float)):
        details.append(f"eval_loss={loss:.4f}")
    error = result.final_metrics.get("error")
    if error:
        details.append(f"error={error}")
    return " ".join(details)
