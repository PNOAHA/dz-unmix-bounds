# Real per-grain detrital-zircon U–Pb age samples (n = 15)

Fifteen real detrital-zircon U–Pb per-grain age samples used by
`make_fig8_real_diagnostic.py` (paper Fig. 8): eleven from Cretaceous strata of the
Songliao Basin and four from Jurassic–Lower Cretaceous strata of the northern North
China Craton (NCC) margin; 1,091 grains in total (31–111 per sample). Each CSV carries
its metadata as `#`-comment headers (sample name, basin, formation, source database,
`source_doi`, grain count) followed by a single `age_Ma` column. (Earlier versions,
v1.1, described all fifteen as "Lower Cretaceous"; that label is not supported for the
NCC-margin samples or for all Songliao formations and is no longer used.)

## Sources and licence

Compiled from published open community datasets and redistributed under their
original terms (CC-BY or equivalent); these data are **not** covered by the MIT licence
of the code. Please cite the original studies (per-sample `source_doi`) when reusing:

- NCC margin: Yang et al. (2006), EPSL 246, 336–352, https://doi.org/10.1016/j.epsl.2006.04.029
  (05FW007, 05FW008); Xu et al. (2016), Int. Geol. Rev. 58, 1528–1556,
  https://doi.org/10.1080/00206814.2016.1168323 (100821-1, 100821-3). Via OneDZ.
- Songliao: Li et al. (2012), Gondwana Res. 22, 943–955, https://doi.org/10.1016/j.gr.2012.04.002
  (SL0929–SL0932; via OneDZ); Wang et al. (2022), Sediment. Geol. 433, 106133,
  https://doi.org/10.1016/j.sedgeo.2022.106133 (R31354-*); Song et al. (2022), Geosci. Front.
  13, 101377, https://doi.org/10.1016/j.gsf.2022.101377 (R31370-*; S12654 reuses data of
  Zhang et al., 2017, https://doi.org/10.1016/j.earscirev.2017.05.015). The R31354/R31370
  records were obtained through the Puetz et al. (2024) global database (Sci. Data 11, 56,
  https://doi.org/10.1038/s41597-023-02902-9); the `source_doi` fields point to the
  original studies, not to the compilation.

## Sample ages (literature lookup, 2026-10-19)

| Sample | Formation | Depositional age (source) |
|---|---|---|
| 05FW007 | Longmen Fm (compiled metadata; link not verified in Yang et al., 2006) | Middle Jurassic in the Jingxi Basin (Hao et al., 2019, https://doi.org/10.1007/s00531-019-01743-5); youngest grain in file 143 Ma |
| 05FW008 | Jiulongshan Fm (compiled metadata; link not verified) | early Late Jurassic in the Jingxi Basin (Hao et al., 2019); youngest grain 142 Ma |
| 100821-1 | lower Tuchengzi Fm (Xu et al., 2016) | Late Jurassic–Early Cretaceous, ≈154–137 Ma (Xu et al., 2012, https://doi.org/10.1016/j.palwor.2012.10.003) |
| 100821-3 | upper Tuchengzi Fm (Xu et al., 2016) | as above |
| SL0929, SL0930, SL0931 | Denglouku Fm (Li et al., 2012) | Cretaceous; youngest zircon of the group 117 ± 3 Ma (Li et al., 2012) |
| SL0932 | Quantou Fm (Li et al., 2012) | Cretaceous; reported youngest zircon 103 ± 5 Ma (not present in this file) |
| R31354-S12518 | lower Denglouku Fm, ZC3.3 (Wang et al., 2022) | Lower Cretaceous (Wang et al., 2022) |
| R31354-S12519 | lower Yingcheng Fm, ZC3.4 (Wang et al., 2022) | Lower Cretaceous |
| R31370-S12650 | Yingcheng Fm, SY8 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12651 | Shahezi Fm, SY13 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12652 | Yingcheng Fm (2nd member), YC2 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12653 | Denglouku Fm (4th member), SLNOR29 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12654 | Denglouku Fm, W902-1 / W-1 (Song et al., 2022) | Lower Cretaceous |

Ages are as stated in the cited literature; youngest-zircon ages are maximum
depositional ages. Known unresolved points (files not altered): grain counts of
100821-1 / 100821-3 (82 / 111) differ from those reported by Xu et al. (2016)
(110 / 81); the 103 ± 5 Ma grain reported for SL0932 is absent here. The same fifteen
samples are analysed by the companion Bayesian study (dz-bayes-uq).
