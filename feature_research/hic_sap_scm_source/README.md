# SAP / SCM source-specified feature research

Isolated from mainline HIC (`EXP-H114` unused).

- Prior fidelity-blocked state: commit `83dfed7d`
- Active SOURCE24 after resolved `positive_sum_mean` and SCM charge
- Artifacts under `results/` and `features/`

## Reproduce

```bash
.venv/bin/python feature_research/hic_sap_scm_source/src/generate_features.py
.venv/bin/python feature_research/hic_sap_scm_source/src/run_pipeline.py
.venv/bin/python -m pytest feature_research/hic_sap_scm_source/tests -q
```
