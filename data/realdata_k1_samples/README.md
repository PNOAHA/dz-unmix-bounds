# Real Lower Cretaceous per-grain age samples (n = 15)

Fifteen real Lower Cretaceous detrital-zircon U-Pb per-grain age samples used by
`make_fig8_real_diagnostic.py` (Fig. 8): eleven from the Songliao Basin and four
from the northern North China Craton margin; 1,091 grains in total (31-111 per
sample). Each CSV carries its metadata as `#`-comment headers (sample name,
basin, formation, source database, `source_doi`, grain count) followed by a
single `age_Ma` column.

Provenance: compiled from published open (CC-BY) community datasets -
North China Craton margin samples from https://doi.org/10.1016/j.epsl.2006.04.029
and https://doi.org/10.1080/00206814.2016.1168323; Songliao samples via the OneDZ
compilation (original study https://doi.org/10.1016/j.gr.2012.04.002) and the
Puetz compilation (https://doi.org/10.1016/j.sedgeo.2022.106133,
https://doi.org/10.1016/j.gsf.2022.101377). Please cite the original sources
(per-sample `source_doi`) when reusing the data. The same fifteen samples are
analysed in full by the companion Bayesian study (dz-bayes-uq).
