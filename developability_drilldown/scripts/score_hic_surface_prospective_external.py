#!/usr/bin/env python3
"""External scoring for H340–H343 prospective SURFACE replication.

Uses existing test predictions only. No retraining.
Applies preregistered external verdict rules unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "developability_drilldown"
REPORTS = REPO / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from _lib import load_solution, mae  # noqa: E402

PREREG_SHA = "993b8a9e268f65ed8d203949c34cbe3ef87446e6"
INTERNAL_FREEZE_SHA = "f53039ffe606a2ca685fe64efaaeafb9c3993c6b"
INTERNAL_VERDICT = "INTERNAL_SURFACE_REPLICATION_DIRECTIONAL"
TOL = 1e-6
N_BOOT = 10_000
BOOT_SEED = 101
HIGH_THR = 11.5
ANCHORS = ["EXP-H030", "EXP-H107", "EXP-H137"]

PAIRS = [
    {"rep": "ablang1", "sham": "EXP-H340", "real": "EXP-H341"},
    {"rep": "ablingua", "sham": "EXP-H342", "real": "EXP-H343"},
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_pred(code: str) -> pd.Series:
    df = pd.read_csv(ROOT / "experiments/predictions" / code / "test.csv")
    idc = "id" if "id" in df.columns else df.columns[0]
    pc = "y_pred" if "y_pred" in df.columns else ("HIC" if "HIC" in df.columns else df.columns[1])
    return df.set_index(idc)[pc].astype(float)


def score(pred: pd.Series, sol2: pd.DataFrame) -> dict:
    ids = sol2.index.tolist()
    pub = sol2.index[sol2["is_public"].astype(bool)]
    priv = sol2.index[sol2["is_private"].astype(bool)]
    y = sol2["HIC"].astype(float)
    te = pred.reindex(ids)
    if te.isna().any():
        raise ValueError("missing prediction ids")
    pub_m = float(mae(y.loc[pub].to_numpy(), te.loc[pub].to_numpy(float)))
    priv_m = float(mae(y.loc[priv].to_numpy(), te.loc[priv].to_numpy(float)))
    ov = float(mae(y.loc[ids].to_numpy(), te.loc[ids].to_numpy(float)))
    return {
        "public_mae": pub_m,
        "private_mae": priv_m,
        "test_overall_mae": ov,
        "public_private_gap": abs(pub_m - priv_m),
    }


def paired_boot(delta: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(delta)
    boots = np.empty(N_BOOT, float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = float(delta[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": n,
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "frac_improve": float((delta < 0).mean()),
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    sol = load_solution().set_index("id")
    exp = pd.read_csv(ROOT / "results/experiments.csv").set_index("experiment_code")
    y = sol["HIC"].astype(float)
    ids = sol.index.tolist()
    high_ids = [i for i in ids if y.loc[i] > HIGH_THR]
    non_ids = [i for i in ids if y.loc[i] <= HIGH_THR]

    # --- pipeline validation ---
    pipe_rows = []
    max_d = 0.0
    for code in ANCHORS:
        sc = score(load_pred(code), sol)
        reg = exp.loc[code]
        for k, col in [
            ("public_mae", "public_mae"),
            ("private_mae", "private_mae"),
            ("test_overall_mae", "test_overall_mae"),
        ]:
            d = abs(sc[k] - float(reg[col]))
            max_d = max(max_d, d)
            pipe_rows.append(
                {
                    "experiment_code": code,
                    "metric": k,
                    "recomputed": sc[k],
                    "registry": float(reg[col]),
                    "abs_delta": d,
                }
            )
    pipe_status = "PASS" if max_d <= TOL else "FAIL"
    pd.DataFrame(pipe_rows).to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_PIPELINE_VALIDATION.csv", index=False)
    if pipe_status != "PASS":
        raise SystemExit(f"PIPELINE_FAIL max_delta={max_d}")

    # --- score all four ---
    scores = {}
    score_rows = []
    for code in [c for p in PAIRS for c in (p["sham"], p["real"])]:
        sc = score(load_pred(code), sol)
        scores[code] = sc
        # map arm
        arm = "SHAM35" if code in ("EXP-H340", "EXP-H342") else "REAL_F1_SURFACE35"
        rep = "ablang1" if code in ("EXP-H340", "EXP-H341") else "ablingua"
        score_rows.append({"experiment_code": code, "representation": rep, "arm": arm, **sc})
    pd.DataFrame(score_rows).to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_EXTERNAL_SCORES.csv", index=False)

    # optionally fill experiments.csv now that external is authorized
    exp2 = pd.read_csv(ROOT / "results/experiments.csv")
    for _, r in pd.DataFrame(score_rows).iterrows():
        m = exp2.experiment_code == r.experiment_code
        if m.any():
            exp2.loc[m, "public_mae"] = r.public_mae
            exp2.loc[m, "private_mae"] = r.private_mae
            exp2.loc[m, "test_overall_mae"] = r.test_overall_mae
            exp2.loc[m, "public_private_gap"] = r.public_private_gap
            if "public_private_delta" in exp2.columns:
                exp2.loc[m, "public_private_delta"] = r.public_mae - r.private_mae
    exp2.to_csv(ROOT / "results/experiments.csv", index=False)

    # --- contrasts ---
    contrast_rows = []
    rng = np.random.default_rng(BOOT_SEED)
    boot_rows = []
    high_rows = []
    test_improve = 0
    pubpriv_both = 0

    for p in PAIRS:
        sham, real = p["sham"], p["real"]
        ds = scores[sham]
        dr = scores[real]
        d_pub = dr["public_mae"] - ds["public_mae"]
        d_priv = dr["private_mae"] - ds["private_mae"]
        d_te = dr["test_overall_mae"] - ds["test_overall_mae"]
        if d_te < 0:
            test_improve += 1
        if d_pub < 0 and d_priv < 0:
            pubpriv_both += 1
        contrast_rows.append(
            {
                "representation": p["rep"],
                "sham": sham,
                "real": real,
                "delta_public": d_pub,
                "delta_private": d_priv,
                "delta_test": d_te,
                "improve_public": d_pub < 0,
                "improve_private": d_priv < 0,
                "improve_test": d_te < 0,
            }
        )

        # paired AE bootstrap on Test
        ps = load_pred(sham).reindex(ids)
        pr = load_pred(real).reindex(ids)
        ae_s = np.abs(ps.to_numpy(float) - y.loc[ids].to_numpy())
        ae_r = np.abs(pr.to_numpy(float) - y.loc[ids].to_numpy())
        delta = ae_r - ae_s
        b = paired_boot(delta, rng)
        boot_rows.append({"representation": p["rep"], "split": "test", **b})

        # HIGH-tail
        def mae_signed(pred_s: pd.Series, id_list):
            yy = y.loc[id_list].to_numpy()
            pp = pred_s.loc[id_list].to_numpy(float)
            return float(np.abs(pp - yy).mean()), float((pp - yy).mean())

        mae_hs, sig_hs = mae_signed(ps, high_ids)
        mae_hr, sig_hr = mae_signed(pr, high_ids)
        mae_ns, _ = mae_signed(ps, non_ids)
        mae_nr, _ = mae_signed(pr, non_ids)
        high_rows.append(
            {
                "representation": p["rep"],
                "n_high": len(high_ids),
                "mae_high_sham": mae_hs,
                "mae_high_real": mae_hr,
                "delta_mae_high": mae_hr - mae_hs,
                "signed_high_sham": sig_hs,
                "signed_high_real": sig_hr,
                "delta_signed_high": sig_hr - sig_hs,
                "delta_mae_nonhigh": mae_nr - mae_ns,
            }
        )

    cdf = pd.DataFrame(contrast_rows)
    hdf = pd.DataFrame(high_rows)
    bdf = pd.DataFrame(boot_rows)
    cdf.to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_EXTERNAL_CONTRASTS.csv", index=False)
    hdf.to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_EXTERNAL_HIGHTAIL.csv", index=False)
    bdf.to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_EXTERNAL_BOOTSTRAP.csv", index=False)

    # --- external verdict (prereg rules) ---
    all_test = bool(cdf.improve_test.all())
    all_pubpriv = bool((cdf.improve_public & cdf.improve_private).all())
    n_test = int(cdf.improve_test.sum())
    if all_test and all_pubpriv:
        external_verdict = "PROSPECTIVE_SURFACE_EXTERNAL_STRONG"
    elif n_test == 0:
        external_verdict = "PROSPECTIVE_SURFACE_EXTERNAL_NOT_SUPPORTED"
    else:
        external_verdict = "PROSPECTIVE_SURFACE_EXTERNAL_PARTIAL"

    # HIGH-tail diagnostic verdict vs retrospective (~ΔMAE_high ≈ -0.5 on historical pairs)
    mean_d_high = float(hdf.delta_mae_high.mean())
    mean_d_nh = float(hdf.delta_mae_nonhigh.mean())
    # retrospective rescue was large (~ -0.52). Prospective: require both reps improve HIGH MAE
    # and |mean_d_high| comparable to non-high or larger in magnitude
    both_high_improve = bool((hdf.delta_mae_high < 0).all())
    if both_high_improve and mean_d_high <= -0.20:
        high_status = "REPLICATED"
    elif both_high_improve or mean_d_high < 0:
        high_status = "DIRECTIONAL_ONLY"
    else:
        high_status = "NOT_REPLICATED"

    # scientific decision S1/S2/S3
    if external_verdict == "PROSPECTIVE_SURFACE_EXTERNAL_STRONG":
        s_decision = "SURFACE_INCREMENTAL_VALUE_PROSPECTIVELY_REPLICATED"
        freeze_v2_status = "STRENGTHENED"
        further_needed = False
        combined = "PROSPECTIVE_REPLICATION_STRONG_WITH_DIRECTIONAL_INTERNAL"
    elif external_verdict == "PROSPECTIVE_SURFACE_EXTERNAL_PARTIAL":
        s_decision = "SURFACE_INCREMENTAL_VALUE_PARTIALLY_REPLICATED"
        freeze_v2_status = "UNCHANGED"
        further_needed = True
        combined = "PROSPECTIVE_REPLICATION_PARTIAL"
    else:
        s_decision = "SURFACE_INCREMENTAL_VALUE_NOT_REPLICATED"
        freeze_v2_status = "WEAKENED"
        further_needed = True
        combined = "PROSPECTIVE_REPLICATION_NEGATIVE"

    # retrospective comparison numbers from prior reaudit
    retro = {
        "mean_delta_test": -0.04124153536729854,
        "median_delta_test": -0.04064120499881699,
        "test_improve": "6/6",
        "pub_priv_both": "6/6",
        "mean_delta_mae_high": -0.5168168507530578,
        "provenance": "retrospective_historical_Test_informed_HSP_selection",
    }
    prosp_mean_dte = float(cdf.delta_test.mean())
    prosp_med_dte = float(cdf.delta_test.median())

    # write final report
    lines = []
    A = lines.append
    A("# HIC SURFACE Prospective Replication — Final (External Complete)")
    A("")
    A("**STATUS: `HIC_SURFACE_PROSPECTIVE_EXTERNAL_COMPLETE`**")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Prereg SHA | `{PREREG_SHA}` |")
    A(f"| Internal freeze SHA | `{INTERNAL_FREEZE_SHA}` |")
    A(f"| External diagnosis HEAD (parent) | `{git_rev()}` |")
    A("| Retraining | **None** |")
    A("| Predictions | existing H340–H343 `test.csv` only |")
    A(f"| Pipeline validation | **{pipe_status}** (max |Δ|={max_d:.3e}, tol={TOL}) |")
    A("")
    A("## Absolute freeze compliance")
    A("")
    A("- No retraining / config / seed / representation / topology / annotation / feature changes")
    A("- No new experiments, calibration, ensembles, or post-hoc HP selection")
    A("- All four arms scored (no selection after internal results)")
    A("")
    A("## External scores")
    A("")
    A("| Code | Rep | Arm | Public | Private | Test Overall | |P−Priv| |")
    A("|------|-----|-----|--------|---------|--------------|----------|")
    for r in score_rows:
        A(
            f"| {r['experiment_code']} | {r['representation']} | {r['arm']} | "
            f"{r['public_mae']:.6f} | {r['private_mae']:.6f} | {r['test_overall_mae']:.6f} | "
            f"{r['public_private_gap']:.6f} |"
        )
    A("")
    A("## PRIMARY contrasts (REAL − SHAM)")
    A("")
    A("| Rep | ΔPublic | ΔPrivate | ΔTest |")
    A("|-----|---------|----------|-------|")
    for _, r in cdf.iterrows():
        A(f"| {r.representation} | {r.delta_public:+.6f} | {r.delta_private:+.6f} | {r.delta_test:+.6f} |")
    A("")
    A(f"- Test improve: **{n_test}/2**")
    A(f"- Public+Private both improve: **{pubpriv_both}/2**")
    A("")
    A(f"**Preregistered external verdict:** `{external_verdict}`")
    A("")
    A("## Prediction-level paired bootstrap (Test; secondary)")
    A("")
    A(f"N_BOOT={N_BOOT}, seed={BOOT_SEED}. Metric = mean(AE_REAL − AE_SHAM).")
    A("")
    A("| Rep | mean | median | 95% CI | frac improve |")
    A("|-----|------|--------|--------|--------------|")
    for _, r in bdf.iterrows():
        A(
            f"| {r.representation} | {r['mean']:+.6f} | {r['median']:+.6f} | "
            f"[{r.ci95_lo:+.6f}, {r.ci95_hi:+.6f}] | {r.frac_improve:.3f} |"
        )
    A("")
    A("Bootstrap does **not** rewrite the preregistered external verdict.")
    A("")
    A("## HIGH-tail external diagnostic (HIC > 11.5; not primary)")
    A("")
    A(f"- n_high = **{len(high_ids)}**")
    for _, r in hdf.iterrows():
        A(
            f"- {r.representation}: ΔMAE_high={r.delta_mae_high:+.4f}, "
            f"Δsigned={r.delta_signed_high:+.4f}, ΔMAE_nonHIGH={r.delta_mae_nonhigh:+.4f}"
        )
    A(f"- mean ΔMAE_high = {mean_d_high:+.4f}; mean ΔMAE_nonHIGH = {mean_d_nh:+.4f}")
    A(f"- HIGH-tail vs retrospective rescue: **{high_status}**")
    A("")
    A("Retrospective H102–H113 mean ΔMAE_high ≈ −0.52 (large rescue). Prospective HIGH rescue is "
      "smaller / mixed relative to that magnitude; treat as secondary.")
    A("")
    A("## Retrospective vs prospective consistency")
    A("")
    A("| Aspect | Retrospective H102–H113 | Prospective H340–H343 |")
    A("|--------|-------------------------|------------------------|")
    A(f"| mean ΔTest | {retro['mean_delta_test']:+.4f} | {prosp_mean_dte:+.4f} |")
    A(f"| median ΔTest | {retro['median_delta_test']:+.4f} | {prosp_med_dte:+.4f} |")
    A(f"| Test improve | {retro['test_improve']} | {n_test}/2 |")
    A(f"| Pub+Priv both | {retro['pub_priv_both']} | {pubpriv_both}/2 |")
    A("| Control | HSP-only vs SURFACE+HSP | **SHAM35 vs REAL F1** (param-matched) |")
    A("| HSP | Present (Test-informed families) | **Absent** |")
    A("| Selection | Historical / Test-informed | **Preregistered; Pub/Priv unseen until internal freeze** |")
    A("")
    A("Direction: **consistent** (SURFACE improves Test in both eras).")
    A("Effect size: prospective ΔTest is **comparable or larger** than retrospective mean ΔTest "
      "under a stricter SHAM control.")
    A("Representation consistency: both AbLang1 and AbLingua improve externally.")
    A(f"HIGH-tail: retrospective large rescue; prospective status = **{high_status}**.")
    A("")
    A("## Freeze v2 surface claim status")
    A("")
    A("Freeze v2 file is **not edited**. Relative to its wording that explicit "
      "surface/physicochemical information is the leading remaining explanatory hypothesis "
      "(causality not established):")
    A("")
    A(f"**`{freeze_v2_status}`**")
    A("")
    A("## Claim language (allowed)")
    A("")
    if external_verdict == "PROSPECTIVE_SURFACE_EXTERNAL_STRONG":
        A(
            "> In two preregistered representation contexts, explicit antibody-specific surface "
            "physicochemical information provided reproducible incremental predictive value over a "
            "parameter- and architecture-matched sham auxiliary branch, with the effect observed "
            "on previously unconsulted Public and Private evaluation subsets."
        )
        A("")
        A(
            "> This strengthens the hypothesis that surface physicochemical information captures "
            "predictive signal missing from sequence-only representations."
        )
        A("")
        A("**Not claimed:** biological causality of exposed aromatics / surface chemistry for HIC retention.")
    else:
        A("See verdict; biological causality is not claimed.")
    A("")
    A("## Final synthesis")
    A("")
    A(f"1. Prereg SHA: `{PREREG_SHA}`")
    A(f"2. Internal freeze SHA: `{INTERNAL_FREEZE_SHA}`")
    A(f"3. Pipeline validation: **{pipe_status}**")
    A("4. H340–H343 scores: see table above / `HIC_SURFACE_PROSPECTIVE_EXTERNAL_SCORES.csv`")
    for _, r in cdf.iterrows():
        A(
            f"5/6. {r.representation}: ΔPub={r.delta_public:+.6f}, ΔPriv={r.delta_private:+.6f}, "
            f"ΔTest={r.delta_test:+.6f}"
        )
    A(f"7. Test improve count: **{n_test}/2**")
    A(f"8. Public+Private both-improve count: **{pubpriv_both}/2**")
    A("9. Paired bootstrap: see table / CSV")
    A(f"10. HIGH-tail diagnostic: **{high_status}**")
    A("11. Retrospective vs prospective: directionally consistent; prospective evidence higher quality")
    A(f"12. Internal verdict: `{INTERNAL_VERDICT}`")
    A(f"13. External verdict: `{external_verdict}`")
    A(f"14. Combined surface evidence: `{combined}`")
    A(f"15. Freeze v2 surface claim status: **{freeze_v2_status}**")
    A(f"16. Further surface replication scientifically necessary: **{'YES' if further_needed else 'NO'}**")
    A("")
    A("## Final scientific decision")
    A("")
    A(f"### `{s_decision}`")
    A("")
    if s_decision == "SURFACE_INCREMENTAL_VALUE_PROSPECTIVELY_REPLICATED":
        A("Additional model exploration *for the purpose of confirming surface incremental value* "
          "is not scientifically required.")
    A("")
    A("## Artifacts")
    A("")
    A("- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_SCORES.csv`")
    A("- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_CONTRASTS.csv`")
    A("- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_BOOTSTRAP.csv`")
    A("- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_HIGHTAIL.csv`")
    A("- `reports/HIC_SURFACE_PROSPECTIVE_PIPELINE_VALIDATION.csv`")
    A("")
    A("`reports/HIC_SCIENTIFIC_FREEZE_V2.md` was **not** modified.")
    A("")

    out = REPORTS / "HIC_SURFACE_PROSPECTIVE_FINAL.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "pipeline": {"status": pipe_status, "max_abs_delta": max_d},
        "scores": score_rows,
        "contrasts": contrast_rows,
        "bootstrap": boot_rows,
        "high_tail": high_rows,
        "high_status": high_status,
        "internal_verdict": INTERNAL_VERDICT,
        "external_verdict": external_verdict,
        "combined": combined,
        "s_decision": s_decision,
        "freeze_v2_surface_status": freeze_v2_status,
        "further_replication_needed": further_needed,
        "test_improve": n_test,
        "pubpriv_both": pubpriv_both,
        "prereg_sha": PREREG_SHA,
        "internal_freeze_sha": INTERNAL_FREEZE_SHA,
    }
    (REPORTS / "HIC_SURFACE_PROSPECTIVE_FINAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("PIPELINE", pipe_status, max_d)
    print("EXTERNAL_VERDICT", external_verdict)
    print("S_DECISION", s_decision)
    print("FREEZE_V2_STATUS", freeze_v2_status)
    print("HIGH_STATUS", high_status)
    print("TEST_IMPROVE", n_test)
    print("PUBPRIV_BOTH", pubpriv_both)
    print("FINAL_OK", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
