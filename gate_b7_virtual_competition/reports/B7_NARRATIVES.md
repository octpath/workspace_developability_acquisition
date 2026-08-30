# B7 Competition Narratives

## TmApp · A_CV_FIRST
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=3.5444 Public=3.7840 Private=3.7716
- S02 SEQ_SIMPLE_Ridge: hyp='Simple sequence descriptors + Ridge.' | CV=3.7530 Public=3.5763 Private=3.6531 ΔPublic=-0.2077 ΔPrivate(reveal)=-0.1186 [TRUE gain]
- S03 BIO_Ridge: hyp='BIO annotations may capture developability-relevant motifs.' | CV=3.2267 Public=3.7336 Private=3.5789 ΔPublic=+0.1574 ΔPrivate(reveal)=-0.0742
- S04 ABLANG2_PCA32_SVR: hyp='AbLang2 antibody PLM embeddings may beat handcrafted descriptors.' | CV=3.3743 Public=3.6448 Private=3.3946 ΔPublic=-0.0888 ΔPrivate(reveal)=-0.1842 [TRUE gain]
- S05 ESM2_PCA64_SVR: hyp='General protein LM (ESM2) as alternative PLM.' | CV=3.3614 Public=3.5249 Private=3.5029 ΔPublic=-0.1199 ΔPrivate(reveal)=+0.1083 [PUBLIC FALSE POSITIVE]
- S06 STRUCT_ElasticNet: hyp='Structure descriptors may add orthogonal signal for TmApp.' | CV=3.5248 Public=3.7762 Private=3.5033 ΔPublic=+0.2513 ΔPrivate(reveal)=+0.0004
- S07 BIO_ABLANG2_Ridge: hyp='Fusing BIO + AbLang2 may combine annotation and representation signal.' | CV=3.0948 Public=4.0355 Private=3.5278 ΔPublic=+0.2592 ΔPrivate(reveal)=+0.0245 [CV FALSE POSITIVE]
- S08 ENSEMBLE_TOP2_CV: hyp='Ensemble of two best CV models should stabilize.' | CV=2.7361 Public=3.4649 Private=3.3518 ΔPublic=-0.5706 ΔPrivate(reveal)=-0.1761 [TRUE gain]
- S09 ENSEMBLE_TOP3_CV: hyp='Expand ensemble to top-3 CV.' | CV=2.7361 Public=3.4649 Private=3.3518 ΔPublic=-0.0000 ΔPrivate(reveal)=+0.0000
- FINAL PICK: ENSEMBLE_TOP3_CV (Private=3.3518); best Private was ENSEMBLE_TOP2_CV (3.3518); regret=0.0000. Reason: CV-first: minimize CV MAE; Public only as tie-break.

## TmApp · B_BALANCED
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=3.5444 Public=3.7840 Private=3.7716
- S02 SEQ_SIMPLE_Ridge: hyp='Simple sequence descriptors + Ridge.' | CV=3.7530 Public=3.5763 Private=3.6531 ΔPublic=-0.2077 ΔPrivate(reveal)=-0.1186 [TRUE gain]
- S03 BIO_Ridge: hyp='BIO annotations may capture developability-relevant motifs.' | CV=3.2267 Public=3.7336 Private=3.5789 ΔPublic=+0.1574 ΔPrivate(reveal)=-0.0742
- S04 ABLANG2_PCA32_SVR: hyp='AbLang2 antibody PLM embeddings may beat handcrafted descriptors.' | CV=3.3743 Public=3.6448 Private=3.3946 ΔPublic=-0.0888 ΔPrivate(reveal)=-0.1842 [TRUE gain]
- S05 ESM2_PCA64_SVR: hyp='General protein LM (ESM2) as alternative PLM.' | CV=3.3614 Public=3.5249 Private=3.5029 ΔPublic=-0.1199 ΔPrivate(reveal)=+0.1083 [PUBLIC FALSE POSITIVE]
- S06 ENSEMBLE_CV_PUBLIC: hyp='Blend best-CV and best-Public models to hedge disagreement.' | CV=3.1567 Public=3.5451 Private=3.5026 ΔPublic=+0.0201 ΔPrivate(reveal)=-0.0003
- S07 ENSEMBLE_TOP2_PUBLIC: hyp='CV-best and Public-best differ; blend them.' | CV=3.2322 Public=3.4949 Private=3.4727 ΔPublic=-0.0502 ΔPrivate(reveal)=-0.0299 [TRUE gain]
- S08 STRUCT_ElasticNet: hyp='Structure descriptors may add orthogonal signal for TmApp.' | CV=3.5248 Public=3.7762 Private=3.5033 ΔPublic=+0.2813 ΔPrivate(reveal)=+0.0306
- FINAL PICK: ENSEMBLE_TOP2_PUBLIC (Private=3.4727); best Private was ABLANG2_PCA32_SVR (3.3946); regret=0.0781. Reason: Balanced: minimize sum of CV-rank and Public-rank.

## TmApp · C_PUBLIC_DRIVEN
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=3.5444 Public=3.7840 Private=3.7716
- S02 SEQ_SIMPLE_Ridge: hyp='Simple sequence descriptors + Ridge.' | CV=3.7530 Public=3.5763 Private=3.6531 ΔPublic=-0.2077 ΔPrivate(reveal)=-0.1186 [TRUE gain]
- S03 BIO_Ridge: hyp='BIO annotations may capture developability-relevant motifs. (Public improved las' | CV=3.2267 Public=3.7336 Private=3.5789 ΔPublic=+0.1574 ΔPrivate(reveal)=-0.0742
- S04 ABLANG2_PCA32_SVR: hyp='AbLang2 antibody PLM embeddings may beat handcrafted descriptors.' | CV=3.3743 Public=3.6448 Private=3.3946 ΔPublic=-0.0888 ΔPrivate(reveal)=-0.1842 [TRUE gain]
- S05 ESM2_PCA64_SVR: hyp='General protein LM (ESM2) as alternative PLM. (Public improved last round — cont' | CV=3.3614 Public=3.5249 Private=3.5029 ΔPublic=-0.1199 ΔPrivate(reveal)=+0.1083 [PUBLIC FALSE POSITIVE]
- S06 ENSEMBLE_PUBLIC_BEST2: hyp='Average the two best Public models so far.' | CV=3.4601 Public=3.4178 Private=3.4921 ΔPublic=-0.1072 ΔPrivate(reveal)=-0.0108 [TRUE gain]
- S07 BIO_ABLANG2_Ridge: hyp='Reinforce Public-best direction via nearby unused variant.' | CV=3.0948 Public=4.0355 Private=3.5278 ΔPublic=+0.6177 ΔPrivate(reveal)=+0.0357 [CV FALSE POSITIVE]
- S08 STRUCT_ElasticNet: hyp='Structure descriptors may add orthogonal signal for TmApp.' | CV=3.5248 Public=3.7762 Private=3.5033 ΔPublic=-0.2592 ΔPrivate(reveal)=-0.0245 [TRUE gain]
- S09 FUSION_ESM2_STRUCT_ENet: hyp='ESM2 + structure fusion.' | CV=3.5801 Public=3.5299 Private=3.3521 ΔPublic=-0.2463 ΔPrivate(reveal)=-0.1512 [TRUE gain]
- FINAL PICK: ENSEMBLE_PUBLIC_BEST2 (Private=3.4921); best Private was FUSION_ESM2_STRUCT_ENet (3.3521); regret=0.1400. Reason: Public-driven: minimize Public MAE; CV as tie-break.

## HIC · A_CV_FIRST
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=0.5189 Public=0.5408 Private=0.5041
- S02 SEQ_SIMPLE_Ridge: hyp='Sequence physchem/composition + Ridge.' | CV=0.5811 Public=0.6695 Private=0.5823 ΔPublic=+0.1287 ΔPrivate(reveal)=+0.0781
- S03 ESM2_PCA64_SVR: hyp='ESM2 embeddings may capture sequence determinants of HIC.' | CV=0.4872 Public=0.5342 Private=0.4674 ΔPublic=-0.1354 ΔPrivate(reveal)=-0.1148 [TRUE gain]
- S04 STRUCT_ElasticNet: hyp='Exposed hydrophobic surface from ESMFold may matter for HIC.' | CV=0.4780 Public=0.5260 Private=0.4735 ΔPublic=-0.0082 ΔPrivate(reveal)=+0.0060 [PUBLIC FALSE POSITIVE]
- S05 ESM2_PHYS_STRUCT_Ridge: hyp='Ridge on ESM2+structure (alternative fusion head).' | CV=0.5882 Public=0.5889 Private=0.5255 ΔPublic=+0.0629 ΔPrivate(reveal)=+0.0520
- S06 FUSION_ESM2_STRUCT_ENet: hyp='ESM2 + structure fusion for HIC.' | CV=0.4852 Public=0.5134 Private=0.4437 ΔPublic=-0.0755 ΔPrivate(reveal)=-0.0818 [TRUE gain]
- S07 PHYS_STRUCT_Huber: hyp='Huber may reduce center-collapse on elevated HIC.' | CV=1.3751 Public=1.0606 Private=1.0545 ΔPublic=+0.5471 ΔPrivate(reveal)=+0.6108
- FINAL PICK: STRUCT_ElasticNet (Private=0.4735); best Private was FUSION_ESM2_STRUCT_ENet (0.4437); regret=0.0298. Reason: CV-first: minimize CV MAE; Public only as tie-break.

## HIC · B_BALANCED
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=0.5189 Public=0.5408 Private=0.5041
- S02 SEQ_SIMPLE_Ridge: hyp='Sequence physchem/composition + Ridge.' | CV=0.5811 Public=0.6695 Private=0.5823 ΔPublic=+0.1287 ΔPrivate(reveal)=+0.0781
- S03 ESM2_PCA64_SVR: hyp='ESM2 embeddings may capture sequence determinants of HIC.' | CV=0.4872 Public=0.5342 Private=0.4674 ΔPublic=-0.1354 ΔPrivate(reveal)=-0.1148 [TRUE gain]
- S04 STRUCT_ElasticNet: hyp='Exposed hydrophobic surface from ESMFold may matter for HIC.' | CV=0.4780 Public=0.5260 Private=0.4735 ΔPublic=-0.0082 ΔPrivate(reveal)=+0.0060 [PUBLIC FALSE POSITIVE]
- S05 ESM2_PHYS_STRUCT_Ridge: hyp='Ridge on ESM2+structure (alternative fusion head).' | CV=0.5882 Public=0.5889 Private=0.5255 ΔPublic=+0.0629 ΔPrivate(reveal)=+0.0520
- S06 ENSEMBLE_CV_PUBLIC: hyp='Blend best-CV and best-Public models to hedge disagreement.' | CV=0.4733 Public=0.5159 Private=0.4583 ΔPublic=-0.0730 ΔPrivate(reveal)=-0.0672 [TRUE gain]
- S07 FUSION_ESM2_STRUCT_ENet: hyp='ESM2 + structure fusion for HIC.' | CV=0.4852 Public=0.5134 Private=0.4437 ΔPublic=-0.0025 ΔPrivate(reveal)=-0.0146 [TRUE gain]
- S08 ENSEMBLE_TOP2_PUBLIC: hyp='CV-best and Public-best differ; blend them.' | CV=0.4727 Public=0.5091 Private=0.4494 ΔPublic=-0.0043 ΔPrivate(reveal)=+0.0057 [PUBLIC FALSE POSITIVE]
- S09 PHYS_STRUCT_Huber: hyp='Huber may reduce center-collapse on elevated HIC.' | CV=1.3751 Public=1.0606 Private=1.0545 ΔPublic=+0.5514 ΔPrivate(reveal)=+0.6051
- FINAL PICK: ENSEMBLE_TOP2_PUBLIC (Private=0.4494); best Private was FUSION_ESM2_STRUCT_ENet (0.4437); regret=0.0057. Reason: Balanced: minimize sum of CV-rank and Public-rank.

## HIC · C_PUBLIC_DRIVEN
- S01 CONST_MEDIAN: hyp='Constant median baseline.' | CV=0.5189 Public=0.5408 Private=0.5041
- S02 SEQ_SIMPLE_Ridge: hyp='Sequence physchem/composition + Ridge.' | CV=0.5811 Public=0.6695 Private=0.5823 ΔPublic=+0.1287 ΔPrivate(reveal)=+0.0781
- S03 ESM2_PCA64_SVR: hyp='ESM2 embeddings may capture sequence determinants of HIC.' | CV=0.4872 Public=0.5342 Private=0.4674 ΔPublic=-0.1354 ΔPrivate(reveal)=-0.1148 [TRUE gain]
- S04 STRUCT_ElasticNet: hyp='Exposed hydrophobic surface from ESMFold may matter for HIC. (Public improved la' | CV=0.4780 Public=0.5260 Private=0.4735 ΔPublic=-0.0082 ΔPrivate(reveal)=+0.0060 [PUBLIC FALSE POSITIVE]
- S05 ESM2_PHYS_STRUCT_Ridge: hyp='Ridge on ESM2+structure (alternative fusion head). (Public improved last round —' | CV=0.5882 Public=0.5889 Private=0.5255 ΔPublic=+0.0629 ΔPrivate(reveal)=+0.0520
- S06 ENSEMBLE_PUBLIC_BEST2: hyp='Average the two best Public models so far.' | CV=0.4747 Public=0.5158 Private=0.4559 ΔPublic=-0.0731 ΔPrivate(reveal)=-0.0696 [TRUE gain]
- S07 PHYS_STRUCT_Ridge: hyp='Reinforce Public-best direction via nearby unused variant.' | CV=0.6733 Public=0.6594 Private=0.6132 ΔPublic=+0.1437 ΔPrivate(reveal)=+0.1574
- S08 FUSION_ESM2_STRUCT_ENet: hyp='ESM2 + structure fusion for HIC.' | CV=0.4852 Public=0.5134 Private=0.4437 ΔPublic=-0.1460 ΔPrivate(reveal)=-0.1696 [TRUE gain]
- S09 PHYS_STRUCT_Huber: hyp='Huber may reduce center-collapse on elevated HIC.' | CV=1.3751 Public=1.0606 Private=1.0545 ΔPublic=+0.5471 ΔPrivate(reveal)=+0.6108
- FINAL PICK: FUSION_ESM2_STRUCT_ENet (Private=0.4437); best Private was FUSION_ESM2_STRUCT_ENet (0.4437); regret=0.0000. Reason: Public-driven: minimize Public MAE; CV as tie-break.
