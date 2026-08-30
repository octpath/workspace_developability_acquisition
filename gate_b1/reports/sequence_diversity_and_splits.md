# Sequence diversity and splits

## Threshold scan (TRIPLE_CORE n=324)

| thr | n_groups | largest | median | singleton_frac |
|-----|----------:|--------:|-------:|---------------:|
| 0.85 | 62 | 230 | 1.0 | 0.85 |
| 0.9 | 180 | 20 | 1.0 | 0.84 |
| 0.95 | 282 | 9 | 1.0 | 0.94 |

**Canonical threshold: 0.9** — default 90% identity connected-components on either chain

## Triple-core splits

### canonical

- N Dev/Public/Private: {'Dev': 181, 'Public': 65, 'Private': 78}
- N groups: {'Dev': 105, 'Public': 47, 'Private': 28}
- Group overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- Exact pair overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- NN similarity Public: {'mean_nn_sim': 0.8429296298828547, 'median_nn_sim': 0.8545454545454545, 'p90_nn_sim': 0.889413823272091, 'max_nn_sim': 0.8990825688073395, 'frac_nn_sim_ge_0.9': 0.0}
- NN similarity Private: {'mean_nn_sim': 0.8455831303580076, 'median_nn_sim': 0.851159570785739, 'p90_nn_sim': 0.8884678041670239, 'max_nn_sim': 0.8981481481481481, 'frac_nn_sim_ge_0.9': 0.0}

### shadow_1

- N Dev/Public/Private: {'Dev': 194, 'Public': 65, 'Private': 65}
- N groups: {'Dev': 109, 'Public': 44, 'Private': 27}
- Group overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- Exact pair overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- NN similarity Public: {'mean_nn_sim': 0.8416062248150179, 'median_nn_sim': 0.8411214953271028, 'p90_nn_sim': 0.8909090909090909, 'max_nn_sim': 0.8981481481481481, 'frac_nn_sim_ge_0.9': 0.0}
- NN similarity Private: {'mean_nn_sim': 0.8615482022540151, 'median_nn_sim': 0.8691588785046729, 'p90_nn_sim': 0.8878504672897196, 'max_nn_sim': 0.8928571428571429, 'frac_nn_sim_ge_0.9': 0.0}

### shadow_2

- N Dev/Public/Private: {'Dev': 187, 'Public': 72, 'Private': 65}
- N groups: {'Dev': 98, 'Public': 34, 'Private': 48}
- Group overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- Exact pair overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- NN similarity Public: {'mean_nn_sim': 0.8363421104793309, 'median_nn_sim': 0.8498368194629877, 'p90_nn_sim': 0.8785046728971962, 'max_nn_sim': 0.8981481481481481, 'frac_nn_sim_ge_0.9': 0.0}
- NN similarity Private: {'mean_nn_sim': 0.8435410338092716, 'median_nn_sim': 0.8504672897196262, 'p90_nn_sim': 0.8908479755538579, 'max_nn_sim': 0.8990825688073395, 'frac_nn_sim_ge_0.9': 0.0}

### shadow_3

- N Dev/Public/Private: {'Dev': 188, 'Public': 65, 'Private': 71}
- N groups: {'Dev': 114, 'Public': 40, 'Private': 26}
- Group overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- Exact pair overlap: {'Dev∩Public': 0, 'Dev∩Private': 0, 'Public∩Private': 0}
- NN similarity Public: {'mean_nn_sim': 0.8442451652632839, 'median_nn_sim': 0.8482142857142857, 'p90_nn_sim': 0.8878504672897196, 'max_nn_sim': 0.897196261682243, 'frac_nn_sim_ge_0.9': 0.0}
- NN similarity Private: {'mean_nn_sim': 0.8474021133456833, 'median_nn_sim': 0.8545454545454545, 'p90_nn_sim': 0.8909090909090909, 'max_nn_sim': 0.8981481481481481, 'frac_nn_sim_ge_0.9': 0.0}

