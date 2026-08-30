# CDR annotation source audit

Competition: **Antibody Developability — TmApp & HIC**  
Package: **1.0** (definitive pre-publication correction)  
Convention: **`AUTHOR_MMC2_IMGT_SEGMENTS`**

## Source artifact

| Item | Value |
|---|---|
| Workbook | `raw/shehata/a05/mmc2.xlsx` (Shehata et al. supplement mmc2) |
| Heavy segment columns | `VH FR1`, `VH CDR1`, `VH FR2`, `VH CDR2`, `VH FR3`, `VH CDR3`, `VH FR4` |
| Light segment columns | `VL FR1`, `VL CDR1`, `VL FR2`, `VL CDR2`, `VL FR3`, `VL CDR3`, `VL FR4` |
| Join key | `Clone name` → competition `id` |

Some mmc2 segments contain IMGT alignment gap characters (`-`).  
**Distributed lengths and reconstruction use gap-stripped segments**, matching the distributed `heavy` / `light` strings.

```text
*_cdr*_length = len(gap_strip(mmc2 CDR segment))
gap_strip(FR1)+…+gap_strip(FR4) == distributed heavy/light
```

## Coverage

| Check | Result |
|---|---|
| Antibodies checked | **324 / 324** |
| CDR comparisons | **1944** (324 × 6) |
| Exact length matches | **1944 / 1944** |
| Heavy full-sequence reconstruction | **324 / 324** |
| Light full-sequence reconstruction | **324 / 324** |
| Mismatches | **none** |

## Extreme values (source-supported)

| Field | min (id) | median | max (id) |
|---|---|---:|---|
| `h_cdr1_length` | 4 (`ADI-47246`) | 9 | 11 (`ADI-45464`) |
| `h_cdr2_length` | 15 (`ADI-45440`) | 17 | 22 (`ADI-45402`) |
| `h_cdr3_length` | 7 (`ADI-46690`) | 15 | 33 (`ADI-47221`) |
| `l_cdr1_length` | 9 (`ADI-47163`) | 12 | 19 (`ADI-47197`) |
| `l_cdr2_length` | 7 (`ADI-37123`) | 7 | 11 (`ADI-45440`) |
| `l_cdr3_length` | 5 (`ADI-47177`) | 9 | 20 (`ADI-47256`) |

### Explicit human-review outliers

| ID | Field | Value | mmc2 gap-stripped segment | Match |
|---|---|---:|---|---|
| `ADI-47246` | `h_cdr1_length` | 4 | `GVLA` | PASS |
| `ADI-47221` | `h_cdr3_length` | 33 | `ARVLTDDYSDNPKDPEVRGQVKPPPENYYGTDV` | PASS |
| `ADI-47197` | `l_cdr1_length` | 19 | `KTSQSVLYRSRSTNKDYLA` | PASS |
| `ADI-47256` | `l_cdr3_length` | 20 | `CSDTGGSARYVFGSSSAPYV` | PASS |
| `ADI-45402` | `h_cdr2_length` | 22 | `YISGSSPPNYSSSPNYADSVKG` | PASS |
| `ADI-47260` | `h_cdr3_length` | 30 | `AKEAGPGDYYDSSGYYPSIARHHYYYGMDV` | PASS |

Unusual lengths are **accepted** because they match author mmc2 segments and reconstruct the distributed sequences. They are **not** target-derived.

## Final status

**PASS**
