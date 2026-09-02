# PREPROCESSING_IMPLEMENTATION_AUDIT

Source of truth: `fusion_closure/scripts/run_fusion_closure.py` (executed Fusion Closure).
Historical result files were **not** modified.

## A. PLM branch (HIC / ESM2 Heavy)

| Item | Actual value |
|------|--------------|
| Raw embedding key | `esm2__H` from `round1_embeddings.npz` |
| Raw dimensionality | **1280** |
| StandardScaler on raw dims | **Yes** (`plm_pca`) |
| Order | **StandardScaler(raw) → PCA** (scale **BEFORE** PCA) |
| PCA `n_components` | **32** (`PCA_N = 32`; capped by `min(32, n_train-1, n_features)`) |
| PCA `whiten` | **False** (sklearn default; not passed) |
| PCA `random_state` | 42 |

```211:217:organizer_extension/feature_prospecting/fusion_closure/scripts/run_fusion_closure.py
def plm_pca(X_plm_tr, X_plm_te, n_comp=PCA_N):
    sc = StandardScaler()
    Ztr = sc.fit_transform(X_plm_tr)
    Zte = sc.transform(X_plm_te)
    n = min(n_comp, Ztr.shape[0] - 1, Ztr.shape[1])
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(Ztr), pca.transform(Zte)
```

## B. AROMATIC branch

| Item | Actual value |
|------|--------------|
| Feature count | **15** canonical (`AROMATIC-TOPO/FEATURE_SPEC.json`) |
| StandardScaler | **Yes** (`physics_scale`) |
| Imputation | train-median fill for non-finite (`fill_nan_train_median`); then scaler |
| log / clip / other transform | **None** in Fusion Closure path |

## C. Fusion concatenation

| Item | Actual value |
|------|--------------|
| Independent block scaling | **Yes**: PLM scaled inside `plm_pca`; AROMATIC scaled in `physics_scale` |
| Concat | `[PLM_PCA32, physics_scaled]` |
| Scale again after concat | **Yes**: `fit_predict` / `select_ridge_alpha` apply another `StandardScaler` on the concatenated design before Ridge/SVR |

```295:301:organizer_extension/feature_prospecting/fusion_closure/scripts/run_fusion_closure.py
def fit_predict(...):
    a = select_ridge_alpha(Xtr, ytr, groups_tr)
    sc = StandardScaler()
    m = Ridge(alpha=a, random_state=0)
    m.fit(sc.fit_transform(Xtr), ytr)
```

## D. CV and preprocessing fit sample counts

- Primary fold file: `virtual_participant/stage0_cv/cv_primary.csv` (Dev N=162)
- Outer: leave-one-fold-out over fold ids {0,1,2,3,4}

### Outer fold sizes

| Outer val fold | N_val | N_outer_train |
|---------------|-------|---------------|
| 0 | 32 | 130 |
| 1 | 33 | 129 |
| 2 | 33 | 129 |
| 3 | 32 | 130 |
| 4 | 32 | 130 |

### Critical implementation fact (PCA / aromatic scaler)

In `nested_oof_fixed`, **PCA and aromatic StandardScaler are fit on the full outer train**
(~129–130 samples), **not** on the inner-train subset.

Inner `GroupKFold` is used only inside `select_ridge_alpha` / `select_svr_params`
on the **already-built** concatenated design matrix (plus a fresh StandardScaler per inner fold).

Because one outer fold is held out, remaining group count = 4, so
`n_splits = min(5, n_unique_groups) = **4**` (not 5).

| Quantity | Fit sample count in executed code |
|----------|-----------------------------------|
| Aromatic scaler (outer model) | outer train ≈ **129–130** |
| ESM2 StandardScaler (outer model) | outer train ≈ **129–130** |
| PCA basis (outer model) | outer train ≈ **129–130** |
| Inner HP selection scaler on concat | inner train ≈ **96–98** |
| Final Ridge scaler on concat | outer train ≈ **129–130** |

### Typical inner sizes (example outer val fold=0, outer train=130, 4-fold GroupKFold)

| Inner | N_inner_train | N_inner_val |
|-------|---------------|-------------|
| 0 | 97 | 33 |
| 1 | 97 | 33 |
| 2 | 98 | 32 |
| 3 | 98 | 32 |

### Direct answers

1. Scalers/PCA for the scored OOF prediction use **~130** outer-train samples.
2. Nested CV does **not** reduce PCA fitting to ~100; PCA stays on ~130. Nested CV only shrinks the **inner** concat-scaler / α-selection fits to ~96–98.
3. At PCA fitting: raw ESM2 has **p=1280**, **n≈130** → **p ≫ n** yes.

