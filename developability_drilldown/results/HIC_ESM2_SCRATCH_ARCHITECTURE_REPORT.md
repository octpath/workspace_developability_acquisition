# HIC ESM-2 / Scratch architecture report

Platform: `DL_FOLDLOCAL_COSINE_V3`. Codes: H054–H081.

| Representation | Architecture | Merge | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ESM2 | ARCH-H0 | h_only | 0.5022 | 0.5130 | 0.5076 | 0.5130 | 0.4834 | 0.4628 | 0.4731 |
| ESM2 | ARCH-1 | concat | 0.5295 | 0.5237 | 0.5266 | 0.5295 | 0.5059 | 0.4641 | 0.4850 |
| ESM2 | ARCH-1 | mean | 0.5155 | 0.5086 | 0.5120 | 0.5155 | 0.4852 | 0.4591 | 0.4722 |
| ESM2 | ARCH-2 | — | 0.5447 | 0.4914 | 0.5180 | 0.5447 | 0.4992 | 0.4771 | 0.4881 |
| ESM2 | ARCH-3 | concat | 0.5152 | 0.5289 | 0.5220 | 0.5289 | 0.4868 | 0.4577 | 0.4722 |
| ESM2 | ARCH-3 | mean | 0.5133 | 0.5191 | 0.5162 | 0.5191 | 0.4926 | 0.4666 | 0.4796 |
| ESM2 | ARCH-4 | concat | 0.5517 | 0.5134 | 0.5326 | 0.5517 | 0.4965 | 0.4580 | 0.4772 |
| ESM2 | ARCH-4 | mean | 0.5331 | 0.4805 | 0.5068 | 0.5331 | 0.4816 | 0.4615 | 0.4716 |
| ESM2 | ARCH-6 | concat | 0.5719 | 0.5175 | 0.5447 | 0.5719 | 0.4853 | 0.4719 | 0.4786 |
| ESM2 | ARCH-6 | mean | 0.5315 | 0.5172 | 0.5244 | 0.5315 | 0.4980 | 0.4795 | 0.4888 |
| ESM2 | ARCH-6G | concat | 0.5798 | 0.5198 | 0.5498 | 0.5798 | 0.4907 | 0.4712 | 0.4809 |
| ESM2 | ARCH-6G | mean | 0.5430 | 0.4971 | 0.5200 | 0.5430 | 0.4809 | 0.4805 | 0.4807 |
| ESM2 | ARCH-8 | concat | 0.5160 | 0.5116 | 0.5138 | 0.5160 | 0.4671 | 0.4656 | 0.4663 |
| ESM2 | ARCH-8 | mean | 0.5457 | 0.5204 | 0.5331 | 0.5457 | 0.4815 | 0.4771 | 0.4793 |
| SCRATCH | ARCH-H0 | h_only | 0.5468 | 0.5254 | 0.5361 | 0.5468 | 0.5156 | 0.5083 | 0.5120 |
| SCRATCH | ARCH-1 | concat | 0.5644 | 0.5071 | 0.5357 | 0.5644 | 0.4825 | 0.4894 | 0.4860 |
| SCRATCH | ARCH-1 | mean | 0.5605 | 0.4912 | 0.5259 | 0.5605 | 0.4723 | 0.5149 | 0.4936 |
| SCRATCH | ARCH-2 | — | 0.5284 | 0.4749 | 0.5017 | 0.5284 | 0.4796 | 0.4800 | 0.4798 |
| SCRATCH | ARCH-3 | concat | 0.5325 | 0.4794 | 0.5060 | 0.5325 | 0.4841 | 0.4825 | 0.4833 |
| SCRATCH | ARCH-3 | mean | 0.5353 | 0.4955 | 0.5154 | 0.5353 | 0.4807 | 0.4745 | 0.4776 |
| SCRATCH | ARCH-4 | concat | 0.5178 | 0.5168 | 0.5173 | 0.5178 | 0.4606 | 0.4682 | 0.4644 |
| SCRATCH | ARCH-4 | mean | 0.5058 | 0.5133 | 0.5095 | 0.5133 | 0.4859 | 0.4920 | 0.4890 |
| SCRATCH | ARCH-6 | concat | 0.5492 | 0.5641 | 0.5567 | 0.5641 | 0.4885 | 0.4981 | 0.4933 |
| SCRATCH | ARCH-6 | mean | 0.5336 | 0.5069 | 0.5202 | 0.5336 | 0.4713 | 0.5032 | 0.4873 |
| SCRATCH | ARCH-6G | concat | 0.5169 | 0.5043 | 0.5106 | 0.5169 | 0.4852 | 0.4895 | 0.4873 |
| SCRATCH | ARCH-6G | mean | 0.5689 | 0.5028 | 0.5359 | 0.5689 | 0.4899 | 0.5414 | 0.5156 |
| SCRATCH | ARCH-8 | concat | 0.5137 | 0.5027 | 0.5082 | 0.5137 | 0.4798 | 0.4908 | 0.4853 |
| SCRATCH | ARCH-8 | mean | 0.5677 | 0.5436 | 0.5556 | 0.5677 | 0.4723 | 0.4879 | 0.4801 |

## Questions

1. ESM-2 vs Scratch representation advantage?
2. Does Scratch approach PLM when architecture is favorable?
3. Is Heavy-only competitive?
4. Does Light-chain information help?
5. Does explicit H/L communication help?
6. Does geometry help HIC?
7. Does HIC prefer a different architecture from TmApp?

