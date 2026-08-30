# Competition Packaging Audit

Competition package status:

Production split:
    GEN_0001_B_20271100

Split frozen:
    PASS

Distribution data:
    PASS

Secret solution:
    PASS

Participant README:
    PASS

Scientific descriptions source-verified:
    PASS (with noted PDF re-fetch limitation — see below)

Scoring dry-run:
    PASS

Secret leakage audit:
    PASS

Deterministic rebuild:
    PASS

Overall:
    READY_FOR_HUMAN_REVIEW


---

## 1. Production split freeze

| Field | Value |
|---|---|
| split_id | `GEN_0001_B_20271100` |
| seed (metadata) | `20271100` |
| construction_method | `B_simulated_annealing` |
| Public N | 81 |
| Private N | 81 |
| Public ID hash | `2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376` |
| Private ID hash | `f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0` |
| Source artifact | `gate_b7_3_principled_split/config/B7_3_RECOMMENDED_SPLIT.json` |
| Manifest | `competition/organizer/SPLIT_MANIFEST.json` |

Checks:

- Explicit ID lists loaded from B7.3 (not regenerated from seed)  
- Hashes match Gate B7.3 authoritative values  
- Public ∩ Private = ∅; union = full Test N=162  
- Same mask for TmApp and HIC  
- HIC MEDIUM Public/Private = **3/3**  
- HIC HIGH Public/Private = **4/3**  

Split selection was **not** reopened.


## 2. Distribution data

| File | N | Columns | SHA-256 |
|---|---:|---|---|
| `data/distribution/dev.csv` | 162 | id, heavy, light, TmApp, HIC | `4514d27cc886e13f393a60aa8c9e671825534865fb65b222d013626418ad077b` |
| `data/distribution/test_features.csv` | 162 | id, heavy, light | `6fa0426e40257ceed96529fbc15526d81ce282d32decb56a9ca24d696f673d86` |
| `data/distribution/sample_submission.csv` | 162 | id, TmApp, HIC | `fb07f0df42bdce9bf7cd2d79e6fbbb98fffaaefae6f2c9c105ad04b421f383e1` |

Target ranges:

| Split | TmApp (°C) | HIC (min) |
|---|---|---|
| Dev | 57.5 – 83.5 | 8.523 – 12.740 |
| Test (secret) | 52.5 – 81.5 | 8.465 – 13.856 |

Sample submission placeholders = Train medians only:

- TmApp median = 69.5 °C  
- HIC median = 9.115 min  

`validate_data.py`: **PASS** (52 checks, 0 errors).


## 3. Secret solution

| File | N | Columns | SHA-256 |
|---|---:|---|---|
| `data/secret/solution.csv` | 162 | id, TmApp, HIC, is_public, is_private | `b5ed604205b5f524cedc752944c6c2de2eabdf2e661c1ee6aa593b79b515d2f7` |

- `is_public XOR is_private` for all rows  
- Public=81 / Private=81  
- IDs match `SPLIT_MANIFEST.json`  


## 4. Participant README scientific content

File: `competition/participant/README.md`

Covers required conceptual progression:

1. Developability definition (no single universal score)  
2. Why biologically active Abs can still be poor drug candidates  
3. Early risk / process-development / CMO–CDMO cost framing  
4. Heavy/light variable-region primer  
5. TmApp intuition + DSF/Fab apparent transition + “apparent” caveat  
6. HIC intuition + hydrophobicity → self-association / aggregation-**related risk** + **not** a direct aggregation assay  
7. Interpretive HIC bands (not classes)  
8. Two independent MAE leaderboards; no combined score  
9. Public/Private behavior; CV guidance without organizer Private secrets  
10. Provenance / CC BY 4.0 attribution  

Automated wording checks for mandatory TmApp and HIC statements: **PASS**.


## 5. Scientific descriptions — source verification

| Claim | Source used |
|---|---|
| Citation / DOI / authors / journal / year | Crossref + EuropePMC metadata under `raw/shehata/` |
| CC BY 4.0 VoR license | Crossref `content-version=vor` → creativecommons.org/licenses/by/4.0/ |
| Column names TmApp (°C), HIC retention time (min) | Gate B1 assay definitions + mmc2-derived tables |
| Fab TmApp terminology | SI `raw/shehata/a05/mmc1.pdf` figure labels |
| DSF / thermal melt method | Gate B4 `assay_scale_audit.md` citing study STAR Methods |
| HIC as hydrophobicity proxy (not aggregation kinetic assay) | Gate B4 assay audit |

Limitation recorded for human review:

- Live Cell Reports PDF re-fetch was blocked (Cloudflare) during packaging; DSF method text relies on the prior in-repo STAR Methods audit rather than a fresh PDF extract in this session.


## 6. Scoring dry-run

`dry_run.py` created an isolated temp workspace containing **only** `data/distribution/` + `participant/` (+ starter `baseline.py` copy).

- No `solution.csv` / `SPLIT_MANIFEST.json` in participant workspace  
- Median + Ridge baselines produced valid submissions  
- Organizer scoring returned finite Public/Private MAE for both tracks  

Example (median baseline):

| Score | Value |
|---|---:|
| TmApp Public MAE | 3.7840 |
| TmApp Private MAE | 3.7716 |
| HIC Public MAE | 0.5349 |
| HIC Private MAE | 0.5101 |

Dry-run status: **PASS**.


## 7. Secret leakage audit

Visible trees audited: `data/distribution/`, `participant/`.

- No Test labels in `test_features.csv`  
- No `is_public` / `is_private` in distribution CSVs  
- No Public/Private ID lists or `GEN_0001…` split_id in participant-facing files  
- No organizer benchmark model/score tokens in README  
- Machine-readable report: `organizer/LEAKAGE_AUDIT.json`  

Result: **PASS**.


## 8. Deterministic rebuild

Two independent rebuilds into temporary directories produced **byte-identical** CSVs matching the packaged hashes for:

- `dev.csv`  
- `test_features.csv`  
- `sample_submission.csv`  
- `solution.csv`  

Result: **PASS**.


## 9. Package tree

```text
competition/
├── PACKAGING_MANIFEST.json
├── data/
│   ├── distribution/
│   │   ├── DATA_DICTIONARY.md
│   │   ├── dev.csv
│   │   ├── sample_submission.csv
│   │   └── test_features.csv
│   └── secret/
│       └── solution.csv
├── organizer/
│   ├── COMPETITION_SPEC.md
│   ├── LEAKAGE_AUDIT.json
│   ├── PACKAGING_AUDIT.md
│   ├── PROVENANCE.md
│   ├── SPLIT_MANIFEST.json
│   └── scripts/
│       ├── baseline.py
│       ├── build_competition_data.py
│       ├── dry_run.py
│       ├── score_submission.py
│       └── validate_data.py
└── participant/
    └── README.md
```


## 10. Remaining human-review items

1. Confirm CC BY 4.0 attribution text / link placement for the public launch page.  
2. Optionally re-verify DSF wording against a freshly downloaded Cell Reports VoR PDF (blocked here).  
3. Platform tie-break policy for exact Private MAE ties (spec leaves final platform wording).  
4. Git commit hash unavailable (`git_commit: null` — workspace has no commits yet); record after first commit.  
5. Do **not** publish external archive until human sign-off.  
6. Production manifests outside `competition/` were **not** modified (per Gate stop condition).


## Decision

**READY_FOR_HUMAN_REVIEW**
