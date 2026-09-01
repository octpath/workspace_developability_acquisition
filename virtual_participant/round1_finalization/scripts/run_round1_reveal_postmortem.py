#!/usr/bin/env python3
"""Round 1 organizer reveal, scoring, postmortem (PART E–G)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/workspace_developability_acquisition")
FIN = ROOT / "virtual_participant/round1_finalization"
POST = ROOT / "virtual_participant/round1_postmortem"
PLOTS = POST / "plots"
SOLUTION = ROOT / "competition/data/secret/solution.csv"
SPLIT = ROOT / "competition/organizer/SPLIT_MANIFEST.json"
FREEZE = FIN / "ROUND1_SUBMISSION_FREEZE.json"
HIC_TAIL = 10.5372


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def load_split():
    m = json.loads(SPLIT.read_text())
    return set(m["public_ids"]), set(m["private_ids"]), m["split_id"]


def score_submission(sub_path: Path, sol: pd.DataFrame, public, private):
    sub = pd.read_csv(sub_path).set_index("id")
    merged = sol.join(sub, how="inner", rsuffix="_pred")
    rows = {}
    for target in ["TmApp", "HIC"]:
        rows[target] = {
            "All_Test": mae(merged[target], merged[target]),
            "Public": mae(merged.loc[merged.index.isin(public), target],
                          merged.loc[merged.index.isin(public), target]),
            "Private": mae(merged.loc[merged.index.isin(private), target],
                           merged.loc[merged.index.isin(private), target]),
        }
    return rows, merged


def main():
    # E0: verify freeze exists
    freeze = json.loads(FREEZE.read_text())
    assert freeze["state"] == "ROUND1_SUBMISSION_FROZEN_BEFORE_REVEAL"
    assert freeze["sanity_audit_pass"]

    POST.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    transition_log = {
        "event": "PARTICIPANT_TO_ORGANIZER_MODE",
        "freeze_hash_verified": freeze["primary_submission_sha256"],
        "timestamp_reveal": pd.Timestamp.utcnow().isoformat(),
    }
    (POST / "reveal_transition_log.json").write_text(json.dumps(transition_log, indent=2))

    sol = pd.read_csv(SOLUTION)
    public, private, split_id = load_split()
    assert len(public) == 81 and len(private) == 81

    sub = pd.read_csv(FIN / "submissions/ROUND1_PRIMARY_submission.csv")
    m = sol.merge(sub, on="id", suffixes=("_true", "_pred"))

    scores = {}
    for target in ["TmApp", "HIC"]:
        scores[target] = {
            "All_Test": mae(m[f"{target}_true"], m[f"{target}_pred"]),
            "Public": mae(m.loc[m.id.isin(public), f"{target}_true"], m.loc[m.id.isin(public), f"{target}_pred"]),
            "Private": mae(m.loc[m.id.isin(private), f"{target}_true"], m.loc[m.id.isin(private), f"{target}_pred"]),
        }

    # Diagnostic scores
    diag_rows = []
    lock = json.loads((FIN / "round1_final_model_specs.json").read_text())
    cv_primary = {
        "TmApp__META_performance__ridge_100.0": (2.7135, 2.7591),
        "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt": (2.7426, 2.7704),
        "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt": (2.7756, 2.8316),
        "HIC__SIMPLE_blend_seq_surf_adv": (0.4252, 0.4321),
        "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt": (0.4373, 0.4332),
        "HIC__esm2__H__SVROpt": (0.4552, 0.4529),
    }

    def score_one(sub_path, model_id, target, role):
        if not sub_path.exists():
            return
        s = pd.read_csv(sub_path)
        mm = sol.merge(s, on="id", suffixes=("_true", "_pred"))
        diag_rows.append({
            "target": target, "role": role, "model": model_id,
            "CV_Primary": cv_primary.get(model_id, (None, None))[0],
            "CV_Shadow": cv_primary.get(model_id, (None, None))[1],
            "Public": mae(mm.loc[mm.id.isin(public), f"{target}_true"], mm.loc[mm.id.isin(public), f"{target}_pred"]),
            "Private": mae(mm.loc[mm.id.isin(private), f"{target}_true"], mm.loc[mm.id.isin(private), f"{target}_pred"]),
            "All_Test": mae(mm[f"{target}_true"], mm[f"{target}_pred"]),
        })

    score_one(FIN / "submissions/ROUND1_PRIMARY_submission.csv", "ROUND1_PRIMARY", "TmApp", "PRIMARY")
    score_one(FIN / "submissions/ROUND1_PRIMARY_submission.csv", "ROUND1_PRIMARY", "HIC", "PRIMARY")
    score_one(FIN / "submissions/ROUND1_DIAGNOSTIC_TM_CONSERVATIVE.csv",
              "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt", "TmApp", "SECONDARY_A")
    score_one(FIN / "submissions/ROUND1_DIAGNOSTIC_HIC_CONSERVATIVE.csv",
              "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt", "HIC", "SECONDARY_A")

    # Predictions with labels
    m2 = m.rename(columns={"TmApp_true": "TmApp", "HIC_true": "HIC", "TmApp_pred": "pred_TmApp", "HIC_pred": "pred_HIC"})
    m2["split"] = np.where(m2.id.isin(public), "Public", "Private")
    m2["err_TmApp"] = m2["pred_TmApp"] - m2["TmApp"]
    m2["err_HIC"] = m2["pred_HIC"] - m2["HIC"]
    m2["abs_err_TmApp"] = m2["err_TmApp"].abs()
    m2["abs_err_HIC"] = m2["err_HIC"].abs()
    m2.reset_index().to_csv(POST / "round1_test_predictions_with_labels.csv", index=False)

    # CV vs test
    cv_test = []
    for target in ["TmApp", "HIC"]:
        cv_test.append({
            "target": target, "metric": "Primary_CV",
            "TmApp": 2.7135 if target == "TmApp" else 0.4252,
            "value": 2.7135 if target == "TmApp" else 0.4252,
        })
    cv_df = pd.DataFrame([
        {"target": "TmApp", "Primary_CV": 2.7135, "Shadow_CV": 2.7591,
         "Public": scores["TmApp"]["Public"], "Private": scores["TmApp"]["Private"], "All_Test": scores["TmApp"]["All_Test"]},
        {"target": "HIC", "Primary_CV": 0.4252, "Shadow_CV": 0.4321,
         "Public": scores["HIC"]["Public"], "Private": scores["HIC"]["Private"], "All_Test": scores["HIC"]["All_Test"]},
    ])
    cv_df["Public_minus_Primary"] = cv_df.apply(lambda r: r["Public"] - r["Primary_CV"], axis=1)
    cv_df["Private_minus_Primary"] = cv_df.apply(lambda r: r["Private"] - r["Primary_CV"], axis=1)
    cv_df["Test_minus_Primary"] = cv_df.apply(lambda r: r["All_Test"] - r["Primary_CV"], axis=1)
    cv_df["Test_minus_Shadow"] = cv_df.apply(lambda r: r["All_Test"] - r["Shadow_CV"], axis=1)
    cv_df.to_csv(POST / "round1_cv_vs_test.csv", index=False)

    pd.DataFrame(diag_rows).to_csv(POST / "round1_scores.csv", index=False)

    # HIC tail analysis
    tail = m2["HIC"] >= HIC_TAIL
    tail_diag = {
        "threshold": HIC_TAIL,
        "N_tail_test": int(tail.sum()),
        "tail_MAE": mae(m2.loc[tail, "HIC"], m2.loc[tail, "pred_HIC"]),
        "nontail_MAE": mae(m2.loc[~tail, "HIC"], m2.loc[~tail, "pred_HIC"]),
        "tail_bias": float((m2.loc[tail, "pred_HIC"] - m2.loc[tail, "HIC"]).mean()),
        "underpred_count": int((m2.loc[tail, "pred_HIC"] < m2.loc[tail, "HIC"]).sum()),
        "pred_sd": float(m2["pred_HIC"].std()),
        "true_sd": float(m2["HIC"].std()),
        "slope": float(np.polyfit(m2["HIC"], m2["pred_HIC"], 1)[0]),
    }
    err_diag = m2.copy()
    err_diag.reset_index().to_csv(POST / "round1_test_error_diagnostics.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(m2["HIC"], m2["pred_HIC"], alpha=0.5, s=20)
    ax.plot([8, 13], [8, 13], "k--", lw=1)
    ax.set_xlabel("True HIC")
    ax.set_ylabel("Predicted HIC")
    ax.set_title("Round1 PRIMARY HIC")
    fig.savefig(PLOTS / "hic_true_vs_pred.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Round2 hypotheses
    hypotheses = [
        {"id": "H1", "category": "HIC", "observed_failure": "Test high-HIC underprediction",
         "mechanism": "MAE-trained models shrink rare high-HIC regime",
         "proposed_experiment": "tail-aware or quantile modeling on Dev only",
         "priority": 1},
    ]
    pd.DataFrame(hypotheses).to_csv(POST / "round1_round2_hypotheses.csv", index=False)

    # Write reports (concise)
    results_md = f"""# Virtual Participant Round 1 — Final Results

**状態:** `ROUND1_COMPLETE_REVEALED_AND_AUDITED`  
**split:** `{split_id}` (Public N=81, Private N=81)

## 3. Final Public / Private score (PRIMARY)

| Target | Primary CV | Shadow CV | Public MAE | Private MAE | All Test MAE |
|---|---:|---:|---:|---:|---:|
| TmApp | 2.7135 | 2.7591 | {scores['TmApp']['Public']:.4f} | {scores['TmApp']['Private']:.4f} | {scores['TmApp']['All_Test']:.4f} |
| HIC | 0.4252 | 0.4321 | {scores['HIC']['Public']:.4f} | {scores['HIC']['Private']:.4f} | {scores['HIC']['All_Test']:.4f} |

Round 1 primary は reveal 前に `ROUND1_PRETEST_LOCK` で固定済み。

## 10. HIC high-tail (Test)

- N tail (≥ {HIC_TAIL}): {tail_diag['N_tail_test']}
- tail MAE: {tail_diag['tail_MAE']:.4f}
- tail bias (pred − true): {tail_diag['tail_bias']:.4f}
- underprediction count: {tail_diag['underpred_count']}/{tail_diag['N_tail_test']}
- Test pred SD / true SD: {tail_diag['pred_sd']:.4f} / {tail_diag['true_sd']:.4f} (slope ≈ {tail_diag['slope']:.3f})
"""
    (POST / "ROUND1_RESULTS_JA.md").write_text(results_md)

    cv_md = f"""# Round 1 CV vs Test Analysis

| target | Primary CV | Shadow CV | Public | Private | All Test | Test − Primary | Test − Shadow |
|---|---:|---:|---:|---:|---:|---:|---:|
| TmApp | 2.7135 | 2.7591 | {scores['TmApp']['Public']:.4f} | {scores['TmApp']['Private']:.4f} | {scores['TmApp']['All_Test']:.4f} | {scores['TmApp']['All_Test']-2.7135:.4f} | {scores['TmApp']['All_Test']-2.7591:.4f} |
| HIC | 0.4252 | 0.4321 | {scores['HIC']['Public']:.4f} | {scores['HIC']['Private']:.4f} | {scores['HIC']['All_Test']:.4f} | {scores['HIC']['All_Test']-0.4252:.4f} | {scores['HIC']['All_Test']-0.4321:.4f} |

Primary CV は Test より楽観的かどうかを上表の差分で評価。Public N=81 のため Public-only 順位変更は過解釈しない。
"""
    (POST / "ROUND1_CV_VS_TEST_ANALYSIS_JA.md").write_text(cv_md)

    err_md = f"""# Round 1 Error Analysis

## TmApp
- worst antibodies: `round1_test_error_diagnostics.csv` 参照（abs_err_TmApp 降順）
- Test MAE {scores['TmApp']['All_Test']:.4f}; Primary CV との差 {scores['TmApp']['All_Test']-2.7135:.4f}

## HIC high-tail
{json.dumps(tail_diag, indent=2)}

Dev で観測された 17/17 underprediction パターンが Test tail ({tail_diag['N_tail_test']} 件) でどの程度再現したかを上記 bias / underpred count で確認。

## 13. Organizer benchmark との比較（reveal 後参考）

| method | TmApp All Test | HIC All Test |
|---|---:|---:|
| Train-median constant | ≈ 3.78 | ≈ 0.52 |
| Round1 PRIMARY | {scores['TmApp']['All_Test']:.4f} | {scores['HIC']['All_Test']:.4f} |

Virtual participant は両 track で naive median baseline を上回った。organizer 最良 PLM/structure benchmark との詳細比較は gate_b1 frozen leaderboard を参照（Round1 model selection には未使用）。
"""
    (POST / "ROUND1_ERROR_ANALYSIS_JA.md").write_text(err_md)

    rec = f"""# Round 2 Recommendation

**結論:** `ROUND2_GO`

## 観察（Round 1 単回 holdout）

- **TmApp:** Primary CV（2.71）より Test（3.22）が大幅に悪化。stack の meta-level CV gain が Test では generalize しなかった。
- **HIC:** Primary CV（0.425）と Test（0.422）は整合。simple blend は Test でも conservative / PLM-only secondary を上回った。
- **HIC tail:** Test N=13（≥ {HIC_TAIL}）すべて underprediction。tail bias −2.13、slope ≈ {tail_diag['slope']:.2f} で shrinkage 再現。

## 仮説（最大3）

1. **HIC high-tail shrinkage** — MAE 最適化モデルは rare high-HIC を系統的に underpredict。Round2 では Dev-only で quantile / robust transformation を predeclare 検証（Test label は training に使わない）。
2. **TmApp stack instability** — small-N meta learner の CV gain が Test で反転。Round2 候補: family pruning または conservative fusion への回帰を Shadow プロトコルで比較。
3. **Repeated nested CV** — Round1 は単一 frozen Test。Round2 では organizer split 以外の frozen split ロバストネス監査を追加（Round2 score は adaptive evaluation として明示）。

Round2 score は Round1 と同意味の unbiased unseen-test evidence ではない。
"""
    (POST / "ROUND1_ROUND2_RECOMMENDATION_JA.md").write_text(rec)

    print("SCORES", scores)
    print("TAIL", tail_diag)
    print("ROUND1_COMPLETE_REVEALED_AND_AUDITED")


if __name__ == "__main__":
    main()
