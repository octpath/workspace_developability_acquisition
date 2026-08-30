---
license_name: cc-by-with-restrictions
license_link: LICENSE.md
tags:
- biology
- protein
- antibody
pretty_name: GDPa1
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: GDPa1_v1.2_20250814.csv
extra_gated_fields:
  First name: text
  Last name: text
  Company: text
  Work email: text
  I want to use this dataset for: text
---

# GDPa1: Antibody developability dataset
Contains the assay data for 242 antibodies across 10 assays as described in our latest preprint, [PROPHET-Ab: A high-throughput platform for biophysical antibody developability assessment to enable AI/ML model training](https://www.biorxiv.org/content/10.1101/2025.05.01.651684v1).

![PROPHET-Ab platform](PROPHET-Ab-figure.png)

## Example usage
Using pandas:
```
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/ginkgo-datapoints/GDPa1/GDPa1_v1.2_20250814.csv")
```
Using Hugging Face datasets:
```
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("ginkgo-datapoints/GDPa1")
```

## Data processing
The main table in this data (`GDPa1_v1.1_20250612.csv`) is an averaged form of the tidy data format. We perform the following averaging:
1. Choose only the first production batch (since production batches differed in their constant regions, and the first production batch contained all 246 antibodies)
2. Average by taking the median across all replicates.

This CSV also contains the following computed 5-fold cross-validation columns:
- random_fold: Randomly assigned folds
- hierarchical_cluster_fold: Hierarchical clustering using pairwise sequence identities computed by MMseqs2
- hierarchical_cluster_IgG_isotype_stratified_fold: The same as hierarchical clustering, while attempting to keep IgG subclass representation uniform
across groups.

This is further described in our preprint, in the "Predictive Model Training" section.
We encourage using `hierarchical_cluster_IgG_isotype_stratified_fold` for reporting results.


## Antibody production
Antibodies were expressed in HEK293F and purified using Protein A chromatography prior to developability assessment for all assays. 
Antibodies tested on DLS-kD went through an additional polishing SEC step. 
A smaller subset of antibodies (20 IgGs) was produced in ExpiCHO and purified using Protein A chromatography. 

## Developability assays
1. Titer by Valita
2. Purity by rCE-SDS
3. Aggregation by SEC
4. Thermostability by nanoDSF and DSF
5. Colloidal stability by SMAC
6. Hydrophobicity by HIC
7. Heparin binding by HAC
8. Self association by AC-SINS
9. Polyreactivity by bead-based method against CHO SMP and ovalbumin
10. Self association by DLS-kD (only performed on 10 antibodies, present in the full datasheet)

## Full Datasheet
Our full datasheet in Excel format contains the following information:
- Definitions of column headers in other datasheets
- Antibody sequences
- Assay data in “tidy data” format with one row per replicate
- Assay data summary statistics with average, standard deviation, and replicates for each assay
- Data for nanodsf vs dsf with the same ramp rate in “tidy data” format
- Prior literature data summarizing prior published results compared with GDPa1 data in the associated preprint

## Changelog
- Monday August 18th: Added AC-SINS data with better curve fitting, and corrected PR score calculation. Added a sheet called "Versioning" in the full Excel file.
- Thursday July 3rd: Added ABodyBuilder3 predicted structures.

## Contact
For more information on this data, see our website at https://datapoints.ginkgo.bio/, or contact us at datapoints@ginkgobioworks.com for specific questions about the data.