# SASA / RASA feature audit

## Frozen parameters

- Implementation: `Bio.PDB.SASA.ShrakeRupley`
- Biopython: 1.88
- probe_radius: 1.4
- n_points: 100
- MaxASA scale: Tien2013/Wilke (frozen in `b1_common.MAX_ASA_TIEN2013`)
- Values: {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}

## RASA

`RASA_i = SASA_i / MaxASA(aa_i)` — values >1 **not clipped**.

## Results

| predictor | N | mean frac RASA>1 | max RASA |
|-----------|--:|-----------------:|---------:|
| ABodyBuilder2 | 400 | 0.000591 | 1.152 |
| ESMFold | 400 | 0.000055 | 1.081 |

Feature families: `STR_*_SASA`, `STR_*_SASA_RASA`, `STR_*_SURFACE_PHYS`, `STR_*_INTERFACE`.
BSA = SASA(VH iso)+SASA(VL iso)−SASA(complex); also reported as BSA/2 by name.
