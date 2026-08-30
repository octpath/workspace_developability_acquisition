# AbLang2 failure analysis

Public Spearman=0.357  Private Spearman=0.603
Public MAE=4.133  Private MAE=3.317

## Spearman by TmApp quartile
Public: {Interval(55.999, 67.0, closed='right'): 0.04576766887262615, Interval(67.0, 70.5, closed='right'): -0.09514884131434694, Interval(70.5, 72.5, closed='right'): 0.14508172784151396, Interval(72.5, 81.5, closed='right'): 0.33637328908733694}
Private: {Interval(52.499, 67.0, closed='right'): 0.37423218603661995, Interval(67.0, 70.0, closed='right'): 0.5398627254913716, Interval(70.0, 73.0, closed='right'): 0.24355774955989942, Interval(73.0, 80.5, closed='right'): 0.3523686152905333}

## Public antibodies with largest |rank error| (top 10)
           true       pred  rank_err vh_family vl_family  PL_combined_germline_distance  H_CDR3_len b_cell_subset donor
id                                                                                                                     
ADI-47313  64.5  74.874157      63.5       VH4       VK1                       0.165459          12    IgG memory   UNK
ADI-47290  73.0  65.714546     -54.5       VH1       VL1                       0.031709          16    IgG memory   UNK
ADI-46689  66.5  74.325012      54.0       VH3       VK3                       0.068656           9    IgM memory   UNK
ADI-45368  67.5  79.110194      52.5       VH2       VK1                       0.015435          14    IgG memory   UNK
ADI-47253  64.5  72.388592      50.5       VH3       VK1                       0.099933          14    IgG memory   UNK
ADI-47179  65.5  72.648096      50.0       VH3       VK1                       0.128882          15    IgG memory   UNK
ADI-47206  72.0  64.373045     -48.0       VH3       VK1                       0.151076          13         LLPCs   UNK
ADI-47104  74.5  67.649240     -45.0       VH3       VK3                       0.094831          17    IgM memory   UNK
ADI-47198  67.5  73.932967      42.5       VH3       VK1                       0.088398          15    IgG memory   UNK
ADI-47222  72.5  66.189806     -41.5       VH4       VK3                       0.141416          20         LLPCs   UNK

corr(|rank_err|, germline_distance) Public: 0.079
