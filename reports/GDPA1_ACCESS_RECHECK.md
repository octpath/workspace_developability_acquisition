# GDPa1 Access Recheck — Gate A0.5

**Date:** 2026-08-28  
**Workspace:** `/workspace_developability_acquisition` only  

---

## Executive classification

**GDPa1 access route class:** `PUBLIC_DOWNLOAD_WITH_FORM` **plus** `INDIVIDUAL_HF_GATE_ONLY` as a parallel path.

There is an official **non-Hugging-Face** acquisition path on `https://datapoints.ginkgo.bio/dataset-access`, but it is **not** a direct anonymous file URL. It is a **contact / HubSpot-backed download request form** that emails “download information.” Unauthenticated Hugging Face file bytes remain **401**.

---

## 1. Is there an official non-Hugging-Face download route?

**Yes.**

Evidence from the live Datapoints SPA (`/assets/index-*.js`) and Supabase `datasets` table:

| Route | What it is |
| ----- | ---------- |
| Datapoints “Open Datasets” / `dataset-access` | Catalog UI with **Download Data** button per dataset |
| Form → edge function `submit-dataset-download` | Collects name, company email, research purpose, terms checkbox |
| Email follow-up | UI copy: *“An email with download information will be sent to you shortly.”* |
| Hugging Face `ginkgo-datapoints/GDPa1` | Parallel gated hub (contact share + accept conditions) |

Homepage and catalog also deep-link Hugging Face (`huggingface.co/ginkgo-datapoints`) as the community open-dataset home, but **Datapoints itself does not expose public `.csv/.xlsx` object URLs** in the `datasets` API rows (`download_url` is null for all entries).

---

## 2. What happens when “Download Data” is followed?

Observed flow (programmatic inspection of SPA; form **not** submitted in this audit):

1. User opens dataset card on `dataset-access` (SPA; all routes share one JS bundle).
2. Clicks **Download Data**.
3. Modal opens: **Download {title}**.
4. Required fields: first/last name, job title, company name, company email; optional research purpose; must check *“I agree to the terms and conditions for using this dataset.”*
5. Modal shows **Dataset License Terms** (dual CC BY-NC / CC BY commercial carve-out; see §6).
6. Submit calls Supabase function `submit-dataset-download` with `dataset_id`, contact fields, `agree_terms`.
7. Success toast promises email with download information.
8. Each antibody dataset row also stores a `hubspot_form_id` (e.g. GDPa1 `ff4ff360-5d6c-448b-8c06-0c5df49234ed`), consistent with CRM-tracked lead capture.

**Redirect chain for HF file attempt (unauthenticated):**

```text
GET https://huggingface.co/datasets/ginkgo-datapoints/GDPa1/resolve/main/GDPa1_v1.2_20250814.csv
→ HTTP 401
→ body: "Access to dataset ... is restricted. You must have access ... and be authenticated"
```

No temporary signed CDN URL was obtained without auth/form.

---

## 3. Can data be downloaded without an individual HF gate?

**Possibly yes, via the Datapoints email form — not verified end-to-end here** (we did not submit personal/company identity).

**No**, for silent/anonymous GET of the known HF filenames.

So: **not** `DIRECT_PUBLIC_DOWNLOAD`. Best label: **`PUBLIC_DOWNLOAD_WITH_FORM`**, with HF as **`INDIVIDUAL_HF_GATE_ONLY`** backup.

---

## 4. Is company/organization access possible?

**Partially / operationally yes via company email on the form; no separate “org SSO bulk API” found.**

- Form explicitly asks for **company name** and **company email**.
- Contact: `datapoints@ginkgobioworks.com` (also on HF README).
- Address shown on site: 27 Dry Dock Ave, Boston MA 02210.
- No public enterprise download portal, org token, or bulk S3 listing discovered in SPA/Supabase public tables.
- Apheris federated consortium text on the marketing site refers to a **different** member-owned AbDev training program — not a substitute for GDPa1 open Excel.

**Recommended:** one authorized employee submits the Datapoints form **and/or** accepts HF terms under a company HF org account; then ask Ginkgo whether one approval covers internal co-workers (see draft inquiry).

---

## 5. Exact current files available?

### Catalog (Datapoints Supabase `datasets`, access date 2026-08-28)

| Slug | Title (short) | Stated size | Format | Notes |
| ---- | ------------- | ----------: | ------ | ----- |
| **gdpa1** | 246 IgGs, 10 assays | **728 KB** | Excel | PROPHET-Ab; matches paper N=246 |
| **gdpa3** | 80 OAS IgGs, **five** assays | **113 KB** | Excel | HIC, SEC, nanoDSF, titer, polyreactivity (CHO) |
| **gdpa2-1** | 18 VHH constructs | 301 KB | Excel | Too small for 2-week competition alone |
| **gdpa4** | 160 bispecifics + 65 GDPa1 IgGs | 832 KB | Excel | Specialized bispecific task |
| **gdpa5** | 160 clinical/non-clinical VHHs | 307 KB | Excel | New in catalog (Aug 2026) |

### Hugging Face GDPa1 (metadata from prior Gate; bytes still gated)

- `GDPa1_v1.2_20250814.csv`
- `GDPa1_v1.2_20250814_full.xlsx` (sequences, tidy replicates, summary stats, versioning)
- `structures/`, `LICENSE.md`, `README.md`

**242 vs 246:** Datapoints catalog consistently advertises **246**. HF README intro still says **242** for the processed CSV. Treat 246 as production/full panel; 242 as processed-table filter — confirm after download.

Downloader script remains: `scripts/download_ginkgo.py` (HF_TOKEN path).

---

## 6. Exact current license? (Datapoints download modal)

Quoted from SPA **Dataset License Terms** (shown before form submit):

> This Dataset is made available under the following licenses:  
> - **For non-commercial use:** CC BY-NC 4.0  
> - **For commercial use** (excluding resale, relicensing, or redistribution of the Dataset itself): CC BY 4.0  
>
> **Use involving the resale, redistribution, or relicensing of the Dataset itself or its contents is not permitted under these licenses.**

HF `LICENSE.md` (prior Gate, when readable) used similar commercial carve-out wording under CC BY 4.0.

Artifact: `raw/ginkgo/a05/license_terms_excerpt.txt`.

---

## 7–8. Internal company R&D and redistribution?

Intended use: **small private internal ML competition; not selling the dataset.**

| Question | Classification | Driving clause |
| -------- | -------------- | -------------- |
| **Q1** Employee/company internal R&D use | **LIKELY_ALLOWED_BUT_AMBIGUOUS** | Commercial use under CC BY is contemplated, *excluding* resale/relicensing/**redistribution of the Dataset itself** |
| **Q2** One downloader shares **original** files with colleagues | **EXPLICIT_PERMISSION_REQUIRED** | “redistribution … of the Dataset itself **or its contents** is not permitted” — internal sharing is redistribution of contents |
| **Q3** Organizer distributes derived `dev.csv` / `test_features.csv` internally | **EXPLICIT_PERMISSION_REQUIRED** | Derived tables are still “contents” of the dataset under a strict reading |
| **Q4** Derived features / embeddings / models internal | **LIKELY_ALLOWED_BUT_AMBIGUOUS** | Not the dataset itself; still get written confirmation |

Do **not** treat as `CLEARLY_ALLOWED` for competition logistics without Ginkgo email confirmation.

---

## 9. Exact permission to ask Ginkgo for

Draft inquiry (do **not** send automatically):

> Subject: Permission for internal technical ML competition using GDPa1/GDPa3  
>
> We are a company R&D team evaluating GDPa1 (and possibly GDPa3) for a **private, internal-only** machine-learning training competition among employees. We will not sell, publicly redistribute, or relicense the dataset.  
>
> Please confirm whether the following are permitted under your Dataset License Terms:  
> 1. Internal use of GDPa1/GDPa3 for commercial-entity research/training.  
> 2. Sharing the downloaded Excel/CSV with named internal participants (or alternatively: each participant must accept terms individually).  
> 3. Distributing **derived** internal competition files (e.g., feature matrix without full assay dump, or train/dev splits) only inside our organization.  
> 4. Whether a single company-email form submission / HF org acceptance covers the team.  
>
> Contact we used / will use: [company email]. Preferred datasets: GDPa1, GDPa3.

---

## 10. Official contact mechanism

- Email: **datapoints@ginkgobioworks.com**
- Web: https://datapoints.ginkgo.bio/contact  
- Per-dataset HubSpot form IDs in Supabase `datasets` table  
- HF dataset contact note on GDPa1 card  

---

## 11. Are GDPa3 / GDPa4 easier to obtain?

**Same access mechanism** (form + likely HF gating). No evidence they are ungated.

| Dataset | Obtainability | Competition fit |
| ------- | ------------- | --------------- |
| GDPa3 | Same form; small Excel (113 KB) | Strong **OOD / Private** for GDPa1; thin as standalone (N=80, 5 assays) |
| GDPa4 | Same form | Bispecific-heavy; higher complexity for a 2-week IgG Fv competition |
| GDPa5 | Same form | 160 VHHs — viable **nanobody** alternative if IgG GDPa1 blocked |
| GDPa2.1 | Same form | N=18 — Class C only |

---

## 12. Can GDPa3 serve as external OOD test for GDPa1?

**Yes — scientifically well motivated**, pending access + license confirmation.

- GDPa1: clinical/approved-biased IgG panel, 10 assays.  
- GDPa3: **OAS-diversity-sampled** natural Abs, five overlapping assay types (HIC, SEC, nanoDSF, titer, polyreactivity).  
- Same PROPHET-Ab platform family (linked mAbs methods paper in catalog).  
- Competition paper already used GDPa3-like holdout historically.

Recommend: train/validate on GDPa1; score Public/Private-style generalization on GDPa3 for shared assays only (do not force missing GDPa1-only assays).

---

## Bottom line

1. **Non-HF route exists:** Datapoints form → email, not anonymous CDN.  
2. **Bytes still blocked** in this environment without form/HF auth.  
3. **Internal competition needs written permission** on redistribution of contents/derived splits.  
4. **GDPa1 remains the top antibody multi-assay target** if access+license cleared; **GDPa3** is the natural OOD companion; **GDPa5** is the best Ginkgo fallback if IgG access stalls.
