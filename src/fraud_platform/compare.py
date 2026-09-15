"""Aggregate runs/*/metrics.json into reports/results.md."""

from __future__ import annotations

import json

from fraud_platform.config import REPORTS_DIR, RUNS_DIR


def main() -> None:
    rows = []
    for path in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        t, c, u = m["test"], m["cost"], m.get("uncertainty", {})
        rows.append({
            "run": m["name"], "model": m["model"], "calibration": m.get("calibration", "none"),
            "threshold": m["threshold"], "pr_auc": t["pr_auc"], "precision": t["precision"], "recall": t["recall"],
            "alerts": t["alerts"], "ece": t["ece"], "brier": t["brier"],
            "p_at_r70": t["precision_at_recall"].get("0.70"),
            "cost": c["test_expected_cost"], "saved": c["test_cost_saved_vs_do_nothing"],
            "cost_ci": u.get("eval_cost_at_chosen"), "thr_ci": u.get("threshold"),
        })
    rows.sort(key=lambda r: r["cost"])
    md = [
        "| run | calibration | t | PR-AUC | precision | recall | alerts | P@R=0.7 | ECE | Brier "
        "| test cost (95% CI) | threshold 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"{r['cost']:.0f} ({r['cost_ci']['p2_5']:.0f}–{r['cost_ci']['p97_5']:.0f})" if r["cost_ci"] else f"{r['cost']:.0f}"
        thr = f"{r['thr_ci']['p2_5']:.4f}–{r['thr_ci']['p97_5']:.4f}" if r["thr_ci"] else "—"
        md.append(f"| {r['run']} | {r['calibration']} | {r['threshold']:.4f} | **{r['pr_auc']:.3f}** | {r['precision']:.3f} | "
                  f"{r['recall']:.3f} | {r['alerts']} | {r['p_at_r70']:.3f} | {r['ece']:.4f} | {r['brier']:.4f} | {ci} | {thr} |")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
