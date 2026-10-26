from __future__ import annotations

import json

from scripts.watch_study import collect_statuses, main, render_dashboard


def test_watch_study_collects_and_renders_statuses(tmp_path, capsys):
    run_dir = tmp_path / "study" / "conditions" / "condition_a" / "runs" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "RUNNING", "step": 25, "message": None}),
        encoding="utf-8",
    )

    rows = collect_statuses(tmp_path / "study")
    dashboard = render_dashboard(tmp_path / "study", rows)

    assert len(rows) == 1
    assert rows[0].condition == "condition_a"
    assert rows[0].run == "seed_0"
    assert "RUNNING" in dashboard
    assert "25" in dashboard

    assert main([str(tmp_path / "study"), "--once"]) == 0
    assert "condition_a" in capsys.readouterr().out
