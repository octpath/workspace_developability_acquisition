# AbLingua Structure-Guided Pooling — 最終報告

## 冒頭回答（必須18問）

1. **TripleAA → residue mapping は確立できたか？**  
   **YES。** 公式 `BioTokenizer`: `>seq<` + sliding 3-mer。token 数 = AA 長。詳細は `ABLINGUA_TOKEN_RESIDUE_MAPPING_AUDIT.md`。

2. **オーバーラップ token の residue 化？**  
   residue `r` = span に `r` を含む **非 special** token の final-layer hidden の平均。

3. **CDR 規約？**  
   **IMGT**（`cdr_sequence_index_imgt.csv` / ANARCI）。324/324 配列一致。マスク: CDR_ALL / FRAMEWORK / CDR3。

4. **RASA / exposed threshold？**  
   ESMFold Fv + Shrake–Rupley + Tien2013 MaxASA。  
   **RASA ≥ 0.20**（`feature_extension` participant default）。324/324 mapped、unresolved=0。

5. **CDR_ALL → PARENT 改善（P∩S）？** **NO**（MIXED: Δ −0.029 / +0.018）

6. **CDR3？** **YES**（Δ **+0.015 / +0.038**、fixed 同値）→ **GUIDED_TRY**

7. **EXPOSED？** **NO**（DROP）

8. **BURIED？** **NO**（DROP）

9. **RASA_WEIGHTED？** **NO**（DROP）

10. **BURIED_WEIGHTED？** **NO**（DROP）

11. **EXPOSED_CDR？** **NO**（MIXED）

12. **CDR_FR_SPLIT？** **NO**（MIXED）

13. **COMBO_1/2/3 最強？** いずれも両側正なし。相対的には COMBO_3 が Shadow のみ微正だが Primary 負。**最強の有効候補は単体 CDR3**。

14. **両側改善した候補？** **PARENT+CDR3 のみ**

15. **fixed-alpha 生存？** **CDR3 は YES**（FREE と同じ Δ）

16. **Heavy/Light 偏り（凍結優先 CDR_ALL）？**  
    H_CDR_ONLY / L_CDR_ONLY とも HL CDR_ALL を上回る両側改善なし。信号は鎖非対称診断では明確でない。

17. **Best frozen endgame recipe**  
    **CURRENT_RECIPE + AbLingua GLOBAL + AbLingua CDR3 pooling**  
    Primary **2.732** / Shadow **2.785**（PARENT 2.747 / 2.823）

18. **Final verdict:** **GUIDED_TRY**  
    （CDR3 のみ modest 両側正。探索拡大はしない。）

---

## Sanity（先行スプリント）

HL vs HL+SEQ: 行列は 2560 vs 2638 で SEQ 連結済み。PCA32 MAE 差 ~1e-15。  
**`SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT`**

## Mapping consistency

RESIDUE_GLOBAL vs saved MASKED_MEAN: cosine median **0.999996**（再重み付けにより非同一だが強く一致）。

## 増分結果（PARENT = CURRENT_RECIPE + GLOBAL）

| candidate | P MAE | S MAE | ΔP / ΔS | fixed-α | verdict |
|-----------|-------|-------|---------|---------|---------|
| PARENT (GLOBAL) | 2.747 | 2.823 | 0 / 0 | — | PARENT |
| +CDR_ALL | 2.776 | 2.804 | −0.029 / +0.018 | same | GUIDED_MIXED |
| +CDR3 | **2.732** | **2.785** | **+0.015 / +0.038** | same | **GUIDED_TRY** |
| +EXPOSED | 2.796 | 2.827 | −0.049 / −0.004 | same | GUIDED_DROP |
| +BURIED | 2.783 | 2.826 | −0.037 / −0.004 | same | GUIDED_DROP |
| +RASA_WEIGHTED | 2.794 | 2.827 | −0.047 / −0.004 | same | GUIDED_DROP |
| +BURIED_WEIGHTED | 2.786 | 2.828 | −0.040 / −0.005 | same | GUIDED_DROP |
| +EXPOSED_CDR | 2.765 | 2.797 | −0.018 / +0.026 | same | GUIDED_MIXED |
| +CDR_FR_SPLIT | 2.783 | 2.813 | −0.036 / +0.010 | same | GUIDED_MIXED |
| COMBO_1 | 2.811 | 2.814 | −0.065 / +0.008 | same | GUIDED_MIXED |
| COMBO_2 | 2.818 | 2.832 | −0.071 / −0.010 | same | GUIDED_DROP |
| COMBO_3 | 2.794 | 2.812 | −0.047 / +0.011 | same | GUIDED_MIXED |

Bootstrap（CDR3, B=10000）は結果 CSV の CI 列を参照（記録用）。

## 結論

構造/CDR で「どの residue を pool するか」を変えても、大半は PARENT を悪化または混合。  
**唯一の両側改善は CDR3 pooling**（modest）。これ以上の pooling 探索は行わない。
