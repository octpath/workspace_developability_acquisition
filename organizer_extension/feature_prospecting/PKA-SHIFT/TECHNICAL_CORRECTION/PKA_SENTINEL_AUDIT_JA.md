# PKA Sentinel Audit（target-blind）

**対象:** 凍結済み `PKA-SHIFT_v1` 成果物（immutable）  
**目的:** PROPKA `predicted_pKa=99.99` が canonical ΔpKa 特徴を汚染したかを、target を見ずに確認する。  
**結論:** **はい。99.99 は実数値 pKa ではなく `cysteine_bridge` sentinel である。**

根拠（PROPKA 3.5.1）:

```text
propka/group.py:
  if self.atom.cysteine_bridge:
      self.pka_value = 99.99
```

運用ルール（補正側）: `residue_type==CYS` かつ `predicted_pKa==99.99` → `is_disulfide_cys=true`。

詳細 JSON: [PKA_SENTINEL_AUDIT.json](PKA_SENTINEL_AUDIT.json)

---

## 1. Sentinel 件数

| Generator | CYS total | sentinel 99.99 | non-disulfide CYS |
|-----------|----------:|---------------:|------------------:|
| ESMFold | 1403 | **1158** | 245 |
| ABB2 | 1403 | **1292** | 111 |
| Boltz2 | 1403 | **1360** | 43 |

抗体あたり disulfide-Cys 数（典型）:

- ESMFold: 多くが 2 / 4 / 6（2=92, 4=209, 6=23）
- ABB2: ほぼすべて 4
- Boltz2: 主に 4、一部 6

---

## 2. Global summaries への寄与

`|ΔpKa|` 総和に占める sentinel 寄与（抗体平均）:

| Generator | mean frac \|Δ\| mass from sentinel |
|-----------|-----------------------------------:|
| ESMFold | **~0.86** |
| ABB2 | **~0.85** |
| Boltz2 | **~0.89** |

つまり v1 の `mean_abs_delta_pKa` / `rms` / `frac_ge_*` 等の global 要約は、数値的にほぼ **ジスルフィド sentinel の個数・配置**に支配されていた。

---

## 3. 汚染された canonical features

同一 25 特徴定義のうち、CYS を含むプールを使うものは sentinel の影響を受け得る。

**常に 90.99（=99.99−9.0）で定数化していた特徴（全 generator）:**

- `max_abs_delta_pKa`
- `max_positive_delta_pKa`
- `buried_max_abs_delta`

**実質汚染（sentinel が要約に入る）:**  
global / CDR / framework / interface / buried の mean・max・frac・差分系の大半。

**sentinel の直接汚染が無い／最小:**

- `acidic_*` / `basic_*`（CYS を含めない）
- `max_negative_magnitude_delta_pKa`（99.99 は正方向）

---

## 4. 領域別 sentinel

Sentinel Cys は主に framework / buried（典型的な保存ジスルフィド）。CDR/interface にも一部存在するが、global 汚染の主因は全分子プールへの巨大 |ΔpKa|=90.99 である。

---

## 5. 含意

Original v1 の empirical ラベル（TmApp `MIXED` 等）は **PROPKA disulfide-sentinel の誤用により technical に妥協**している。

Limitation ラベル:

`TECHNICALLY_COMPROMISED_BY_PROPKA_DISULFIDE_SENTINEL`

補正は feature 再選択ではなく、**software output semantics の修正**として同一 25 定義を再計算する。
