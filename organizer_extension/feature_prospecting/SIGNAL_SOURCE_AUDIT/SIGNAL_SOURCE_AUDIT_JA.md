# Signal-Source Audit（Gate 2L）

**性質:** `POST_COMPETITION_DIAGNOSTIC`  
**目的:** AROMATIC-TOPO / POLAR-SAT の明瞭な物理 signal が、構造特異的か配列組成 proxy かを分解する。  
**不変:** 歴史的 family verdict は変更しない。

化学カテゴリ（配列のみ・凍結）:
- donor-capable: `HKNQRSTWY`
- acceptor-capable: `DEHNQSTY`
- polar uncharged: `NQSTWY`
- charged: `DEHKR`

Generator: ESMFold（構造 family の primary と整合）

---

## AROMATIC-TOPO × HIC

| Model | Metrics |
|-------|---------|
| sequence-only YWF | MAE CV/Pub/Pri **0.5406 / 0.5642 / 0.4928** · stand=TEST_ONLY_POSTHOC |
| AROMATIC-TOPO | MAE CV/Pub/Pri **0.5038 / 0.4919 / 0.4661** · stand=REPRODUCIBLE |
| sequence + AROMATIC | MAE CV/Pub/Pri **0.4991 / 0.4943 / 0.4597** · stand=REPRODUCIBLE |

combo − seq ΔMAE CV/Pub/Pri: **-0.0415 / -0.0699 / -0.0332**

**分類:** `GEOMETRY_ADDS_SIGNAL`

解釈: 歴史的 verdict `PROMISING_BUT_REDUNDANT` は不変。本診断は「3D 芳香族幾何が HIC を引き起こす」と断定するためのものではなく、配列組成との分離可能性を測る。

---

## POLAR-SAT × TmApp

| Model | Metrics |
|-------|---------|
| sequence-only polar | MAE CV/Pub/Pri **3.2918 / 3.5784 / 3.9026** · stand=WEAK |
| POLAR-SAT | MAE CV/Pub/Pri **3.2351 / 3.7573 / 3.4320** · stand=REPRODUCIBLE |
| sequence + POLAR-SAT | MAE CV/Pub/Pri **3.2834 / 3.5975 / 3.4718** · stand=REPRODUCIBLE |

combo − seq ΔMAE CV/Pub/Pri: **-0.0085 / 0.0190 / -0.4308**

**分類:** `UNRESOLVED`

解釈: 歴史的 verdict `PROMISING_BUT_REDUNDANT` は不変。埋没未充足極性の構造情報が配列組成を超えるかを評価。

---

## 直答

- AROMATIC は幾何特異的か？ → **GEOMETRY_ADDS_SIGNAL**
- POLAR-SAT は構造特異的か？ → **UNRESOLVED**
