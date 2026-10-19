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
| 05FW007 | Longmen Fm (Yang et al., 2006, Supplementary Table 1) | late Middle Jurassic in the Jingxi Basin (Hao et al., 2019, https://doi.org/10.1007/s00531-019-01743-5); youngest tabulated 206Pb/238U age 143 ± 3 Ma (1σ), unresolved conflict with the regional assignment |
| 05FW008 | Jiulongshan Fm (Yang et al., 2006, Supplementary Table 1) | early Late Jurassic in the Jingxi Basin (Hao et al., 2019); youngest tabulated 206Pb/238U age 142 ± 2 Ma (1σ), unresolved conflict with the regional assignment |
| 100821-1 | lower Tuchengzi Fm (Xu et al., 2016, Section 4.3; see note on labels) | Late Jurassic–Early Cretaceous, ≈154–137 Ma (Xu et al., 2012, https://doi.org/10.1016/j.palwor.2012.10.003) |
| 100821-3 | upper Tuchengzi Fm (Xu et al., 2016, Section 4.3; see note on labels) | as above |
| SL0929, SL0930, SL0931 | Denglouku Fm (Li et al., 2012) | Cretaceous; youngest zircon of the group 117 ± 3 Ma (Li et al., 2012) |
| SL0932 | Quantou Fm (Li et al., 2012) | Cretaceous; youngest zircon 103 ± 5 Ma reported in the text of Li et al. (2012, Section 4.1.2) but not identifiable in their Supplementary Table S1 (75 SL0932 analyses, youngest tabulated 206Pb/238U age 125 ± 3 Ma) |
| R31354-S12518 | lower Denglouku Fm, ZC3.3 (Wang et al., 2022) | Lower Cretaceous (Wang et al., 2022) |
| R31354-S12519 | lower Yingcheng Fm, ZC3.4 (Wang et al., 2022) | Lower Cretaceous |
| R31370-S12650 | Yingcheng Fm, SY8 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12651 | Shahezi Fm, SY13 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12652 | Yingcheng Fm (2nd member), YC2 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12653 | Denglouku Fm (4th member), SLNOR29 (Song et al., 2022) | Lower Cretaceous |
| R31370-S12654 | Denglouku Fm, W902-1 / W-1 (Song et al., 2022) | Lower Cretaceous |

Ages are as stated in the cited literature; youngest-zircon ages are maximum
depositional ages.

Record corrections in v1.2.1 (checked against the original supplementary tables and
the OneDZ input files, Zenodo version https://doi.org/10.5281/zenodo.15522949):
(i) 05FW007: two analyses (spot 12, 206 Ma; spot 17, 296 Ma) that had been dropped
by a cross-sample de-duplication error during extraction were restored (57 -> 59);
both are present in Yang et al. (2006), Supplementary Table 1, and in OneDZ.
(ii) 100821-1 / 100821-3: OneDZ attaches the first analysis of the 100821-3 table to
100821-1 and the first two analyses of the following sample (110909-1) to 100821-3.
The arrays were rebuilt within the table boundaries of the original supplementary
U-Pb table of Xu et al. (2016) under its printed sample labels (82 -> 81 and
111 -> 110). Xu et al. (2016, Section 6.3.1) report the 81 / 110 analysis counts with
the opposite labels, so the association of each array with the lower/upper
Tuchengzi Formation remains unresolved.
(iii) SL0932: unchanged (75 analyses = OneDZ Best Age values); the 103 ± 5 Ma
analysis quoted in the text of Li et al. (2012) is not in their public table or in
OneDZ, and is not added here.
The ages are compiled analyses (OneDZ Best Age), not a uniformly
concordance-filtered set. Total: 1091 analyses, 31-110 per sample. The same fifteen
samples are analysed by the companion Bayesian study (dz-bayes-uq).
