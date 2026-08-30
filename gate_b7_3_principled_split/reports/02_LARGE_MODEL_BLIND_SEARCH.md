# 02 — Large Model-Blind Search

- Starts: A=400, B=400, C=80, total=880
- Valid unique masks: **880**
- Runtime: **1317.6s**
- Best L1=8.000 (GEN_0000_A_20261007; method=A_greedy_swap)
- CAND_12528 L1=26.000; percentile standing (%% generated worse): 1.0

## Method contribution

- A valid packs ingested: 400
- B valid packs ingested: 400
- C valid packs ingested: 80

![landscape](../plots/premodel_lexicographic_landscape.png)

| split_id | method | L1 | L2 | L3 | hic_med_pub | hic_high_pub |
| --- | --- | --- | --- | --- | --- | --- |
| GEN_0000_A_20261007 | A_greedy_swap | 8.0 | 0.23287325596398628 | 0.32510288065843623 | 4 | 3 |
| GEN_0001_B_20271100 | B_simulated_annealing | 8.0 | 0.2343837350460705 | 0.3004115226337449 | 3 | 4 |
| GEN_0002_B_20270923 | B_simulated_annealing | 8.0 | 0.2419847287153954 | 0.2016460905349794 | 3 | 4 |
| GEN_0003_A_20260938 | A_greedy_swap | 8.0 | 0.2529347618275284 | 0.1522633744855967 | 4 | 4 |
| GEN_0004_B_20271225 | B_simulated_annealing | 8.0 | 0.25510473394655236 | 0.2090374648693376 | 3 | 4 |
| GEN_0005_A_20261232 | A_greedy_swap | 8.0 | 0.2577644746851537 | 0.2098765432098765 | 3 | 4 |
| GEN_0006_A_20261270 | A_greedy_swap | 8.0 | 0.26491724655236265 | 0.2583557547671029 | 3 | 3 |
| GEN_0007_B_20271110 | B_simulated_annealing | 8.0 | 0.2673541279103572 | 0.22854387656702022 | 3 | 4 |
| GEN_0008_B_20270925 | B_simulated_annealing | 8.0 | 0.270906254827852 | 0.2966269841269841 | 3 | 4 |
| GEN_0009_B_20271025 | B_simulated_annealing | 8.0 | 0.2848896313663166 | 0.20417519335761758 | 3 | 4 |
| GEN_0010_B_20270922 | B_simulated_annealing | 8.0 | 0.2882181878835974 | 0.19341563786008228 | 3 | 4 |
| GEN_0011_A_20261282 | A_greedy_swap | 8.0 | 0.296844827079978 | 0.16449207828518173 | 4 | 3 |
| GEN_0012_A_20260923 | A_greedy_swap | 8.0 | 0.29870815586167176 | 0.2961309523809524 | 3 | 3 |
| GEN_0013_A_20260905 | A_greedy_swap | 8.0 | 0.30129440631164095 | 0.21810699588477359 | 3 | 3 |
| GEN_0014_A_20261275 | A_greedy_swap | 8.0 | 0.30459844452336876 | 0.23456790123456783 | 2 | 4 |
