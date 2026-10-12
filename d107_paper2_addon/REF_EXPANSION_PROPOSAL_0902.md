## 0. 一句话结论

- **现有 13 条**全部经 Crossref/DataCite 实时复核：**11 条 OK、1 条 NO_DOI（Pan, submitted，按设计无 DOI）、1 条 MISMATCH（Liu et al., 2017）**。MISMATCH 修正：`Liu, Y.-J.` → `Liu, Y.`（Crossref given name "Yongjiang"，无连字符），标题 `...in the eastern part of the Central Asian Orogenic Belt` → `...in the eastern part of Central Asian Orogenic Belt`（注册标题无 "the"）。另有两处风格点（Li 2021 / Xiao & Santosh 2014 漏期号）和一处版本提示（Pan, in preparation 引的是 Zenodo v1.0 版本 DOI 20768132，concept 20768131 下已有 v1.1 = 21739169）。
- **建议新增 must + should 共 25 条**（must 16、should 9），全部 DOI 均已由 verifier 实时核过；采纳后 References 从 **13 → 38 条**。另列 19 条 nice（可选，全部采纳会到 57 条，不建议）。
- 最要紧的不是"多引"，而是**四处结构性缺口**：① 五篇原始数据源论文只以裸 DOI 出现在 Data availability、不在 References；② "OneDZ compilation" / "Puetz compilation" 标签**贴错了 DOI**（贴的是原始研究 Li 2012 / Wang 2022 / Song 2022，两个数据库本身从未被引）；③ §1 L75 把 Sundell & Saylor 2017 说成 bootstrap CI 与其实际方法（inverse Monte Carlo）不符，且 "usually" 被 Tipton et al. 2022 的贝叶斯先例削弱；④ §2 S2 用 Xiao & Santosh 2014（**西段** CAOB）支撑 Solonker 缝合带（东段）。
- ⚠ 顺带发现一个 **data-vs-claim 问题**（非引用问题，但投稿前必须处理）：verifier 读到本地 CSV header 显示 05FW007 = Longmen Fm、05FW008 = Jiulongshan Fm、100821-1/-3 = Tuchengzi Fm，而 Xu et al. 2016 摘要把 Longmen 定为早–中侏罗世、Tuchengzi 为晚侏罗–早白垩世；稿件通篇称"fifteen Lower Cretaceous (K1) samples"，对 NCC 四个样本可能不成立。见 §4。

## 1. 现有 13 条复核

| key | status | 问题 | 修正 |
|---|---|---|---|
| Foreman-Mackey et al., 2013 | OK | 无。Crossref 标题里 emcee 为 `<tt>` 等宽，稿件正体 | 可选：emcee 改斜体/等宽与正文一致 |
| Goodman & Weare, 2010 | OK | 无 | — |
| Huang et al., 2024 | OK | 冒号后 "insights" 小写为 sentence-case 惯例，非错误 | — |
| Li et al., 2021 | OK | Crossref 有期号 4，稿件漏；与 Foreman-Mackey/Goodman/Sundell/Zhao 2001 给期号的做法不一致 | `International Journal of Earth Sciences 110 (4), 1233–1263.`（或全稿统一删期号） |
| Liu et al., 2017 | **MISMATCH** | ① 首作者缩写 `Y.-J.` 与 Crossref "Yongjiang" 不符；② 标题多了 "the" | `Liu, Y., Li, W., Feng, Z., Wen, Q., Neubauer, F., Liang, C., 2017. A review of the Paleozoic tectonics in the eastern part of Central Asian Orogenic Belt. Gondwana Research 43, 123–148. doi:10.1016/j.gr.2016.03.013` |
| Pan, submitted | NO_DOI | 按设计无 DOI；状态词 "submitted" 出现两次（key + 条目内） | 可选：状态只写一次，如 `Pan, L., 2026. Title. Computers & Geosciences, submitted for publication. Preprint: ...` |
| Pan, in preparation | OK | Zenodo 20768132 = v1.0 版本 DOI；concept 20768131 已有 v1.1（21739169）。正文 L432 内联同一 DOI（平行副本） | 若引冻结版：`Code (v1.0): doi:10.5281/zenodo.20768132 (concept doi:10.5281/zenodo.20768131)`；若改 concept，L432 同步改。"in preparation" 为 09-02 用户拍板，不动 |
| Sharman & Johnstone, 2017 | OK | 无 | — |
| Sundell & Saylor, 2017 | OK | 条目无误；但 **L75 对它的描述有误**（见 §2-B） | 改正文，不改条目 |
| Wu et al., 2022 | OK | 无 | — |
| Xiao & Santosh, 2014 | OK | Crossref 有期号 4，稿件漏；且它是**西段** CAOB 综述，用于 L122 东段 Solonker 论述不对口 | `Gondwana Research 25 (4), 1429–1444.`；L122 加 Xiao et al. 2003（见 §2-G） |
| Zhao, 2001 | OK | 无 | — |
| Zhao et al., 2012 | OK | "key" 小写为 sentence-case 惯例 | — |

正文–列表交叉：无孤儿条目、无缺失条目、年份全部一致；`et al.` 用法正确；两作者用 `&`（Elsevier 范例用 "and"，按目标期刊指南全局统一）。L186 `(Goodman & Weare, 2010; emcee, Foreman-Mackey et al., 2013)` 把软件名嵌进引用组，建议改 `(Goodman & Weare, 2010), as implemented in emcee (Foreman-Mackey et al., 2013)`。

## 2. 建议新增（按插入点分组，按稿件顺序）

> 约定：**CURRENT** 为稿件原句（逐字），**PROPOSED** 为最小改动。标 **PARTIAL** 的表示 verifier 判定文献只部分支撑该句，已按其建议微调措辞。单个括号不超过 3 条引用。

### A. §1 L66 — 开篇句 · must

- CURRENT: `Detrital-zircon U–Pb geochronology has become a primary tool for sediment provenance analysis, and a natural way to interpret a measured age spectrum is`
- PROPOSED: `Detrital-zircon U–Pb geochronology has become a primary tool for sediment provenance analysis (Gehrels, 2014), and a natural way to interpret a measured age spectrum is`
- 引用：Gehrels, 2014 — must, SUPPORTS

### B. §1 L75 — 先例与不确定度报告方式 · must（PARTIAL，需改写）

- CURRENT: `Where uncertainty is reported, it is usually a bootstrap confidence interval on the weights (Sundell & Saylor, 2017); we extend this in three ways: ... — a decomposition a single bootstrap interval does not provide.`
- PROPOSED: `Where uncertainty is reported at all, it is usually a Monte-Carlo or resampling spread on the weights (Sharman & Johnstone, 2017; Sundell & Saylor, 2017) or, more rarely, a nonparametric Bayesian posterior (Tipton et al., 2022); we extend this in three ways: ... — a decomposition a single resampling interval does not provide.`
- 引用：Tipton et al., 2022 — must, **PARTIAL**（它是反例/先例，不是支撑原句；原句 "usually a bootstrap CI" 对 Sundell & Saylor 2017 的描述不准确——该文是 inverse Monte Carlo 最佳拟合范围）
- 作者待办：核 Sharman & Johnstone 2017 方法节是否确用 bootstrap，再决定 "resampling" 措辞。

### C. §1 L77 — 弱可识别性 · nice

- CURRENT: `their relative contributions can be only weakly identifiable even when their sum is well constrained, and a point estimate conceals this.`
- PROPOSED: `their relative contributions can be only weakly identifiable even when their sum is well constrained — a classical property of finite-mixture likelihoods (Redner & Walker, 1984) — and a point estimate conceals this.`
- 引用：Redner & Walker, 1984 — nice, PARTIAL

### D. §1 L83–84 — 贝叶斯反问题框架 · must

- CURRENT: `We address this by casting three-end-member detrital-zircon unmixing as a Bayesian inference problem and propagating the full posterior uncertainty.`
- PROPOSED: `We address this by casting three-end-member detrital-zircon unmixing as a Bayesian inference problem (Mosegaard & Tarantola, 1995; Sambridge & Mosegaard, 2002) and propagating the full posterior uncertainty.`
- 引用：Mosegaard & Tarantola, 1995 — must, SUPPORTS；Sambridge & Mosegaard, 2002 — must, SUPPORTS
- nice 备选（同一括号第 3 位，或放 §3.4 首句）：Jasra et al., 2006（地质年代学中贝叶斯 MCMC 混合模型先例；措辞须为 "Bayesian mixture modelling of age distributions" 而非 "end-member unmixing"）；Tarantola, 2005（PARTIAL，仅作一般反问题专著）。

### E. §2 L103–106 — 松辽盆地组 · must + should

- CURRENT: `a North-China-Craton north-margin group and a Songliao Basin group (Fig. 1; coordinates and grain counts in Table 1).`
- PROPOSED: `a North-China-Craton north-margin group and a Songliao Basin group (Feng et al., 2010; Wang et al., 2016; Fig. 1; coordinates and grain counts in Table 1).`
- 引用：Feng et al., 2010 — must, SUPPORTS（松辽构造地层格架，Denglouku 起始后裂陷）；Wang et al., 2016 — should, **PARTIAL**（仅支撑地层/构造背景，不支撑 Fig. 1 的 "wide-rift" 标签——该文把松辽定为 inverted active continental margin basin）
- 说明：§2 目前**没有任何松辽盆地专属文献**，而 15 个样本中 11 个来自松辽。

### F. §2 S1 L115–116 — 早白垩世岩浆作用范围 · nice

- CURRENT: `is widespread across NE Asia (arc-magmatic peak ≈ 124–133 Ma; Li et al., 2021) and contributes a young, narrow age mode.`
- PROPOSED: `is widespread across NE Asia (Xu et al., 2013; Wu et al., 2011), with an arc-magmatic peak ≈ 124–133 Ma (Li et al., 2021), and contributes a young, narrow age mode.`
- 引用：Xu et al., 2013 — nice, SUPPORTS；Wu et al., 2011 — nice, PARTIAL（支撑 ~130 Ma 花岗岩年龄众数；但其结论是古生代花岗岩"没有以前认为的那么广"，**不要**用它支撑 L121 "broad late-Palaeozoic source"）

### G. §2 S2 L118–122 — CAOB 与 Solonker 缝合带 · should（修正 inconsistency 3）

- CURRENT: `the accretionary and arc terranes of the CAOB — the world's largest accretionary orogen, amalgamated through the Palaeozoic with final closure of the Palaeo-Asian Ocean along the Solonker Suture near the northern NCC margin — provide a broad late-Palaeozoic source (Liu et al., 2017; Xiao & Santosh, 2014).`
- PROPOSED: `the accretionary and arc terranes of the CAOB — the world's largest accretionary orogen (Windley et al., 2007), amalgamated through the Palaeozoic with final closure of the Palaeo-Asian Ocean along the Solonker Suture near the northern NCC margin (Xiao et al., 2003) — provide a broad late-Palaeozoic source (Liu et al., 2017; Xiao & Santosh, 2014).`
- 引用：Xiao et al., 2003 — should, SUPPORTS（东段 CAOB / Solonker 原始论文，Crossref 无页码，用 article-number 2002TC001484；印刷引用号 1069 需作者核 PDF）；Windley et al., 2007 — nice, SUPPORTS
- nice 备选：Wilde, 2015（可与 Xiao 2003 同括号）。
- 注意：保留 Xiao & Santosh 2014 在末括号，否则它成孤儿条目；若作者决定删它，References 同步删。

### H. §2 S3 L124–126 — NCC ~1.85 Ga 拼合 · should

- CURRENT: `final Palaeoproterozoic amalgamation along the Trans-North China Orogen (Zhao, 2001; Zhao et al., 2012), a source separated`
- PROPOSED: `final Palaeoproterozoic amalgamation along the Trans-North China Orogen (Zhao, 2001; Zhao et al., 2005, 2012), a source separated`
- 引用：Zhao et al., 2005 — should, SUPPORTS（摘要原文 "collision at about 1.85 Ga ... final amalgamation of the North China Craton"）

### I. §3.1 L149–152 — 单纯形与边缘化似然 · nice

- CURRENT: `live on the 2-simplex (Σ w_k = 1). Rather than assigning grains to sources, we marginalise over the latent source label of each zircon, giving the per-grain mixture likelihood`
- PROPOSED: `live on the 2-simplex (Σ w_k = 1; Aitchison, 1986). Rather than assigning grains to sources, we marginalise over the latent source label of each zircon (McLachlan & Peel, 2000; Frühwirth-Schnatter, 2006), giving the per-grain mixture likelihood`
- 引用：Aitchison, 1986 — nice, SUPPORTS（⚠ Springer DOI 目前 404，见 §3 条目）；McLachlan & Peel, 2000 — nice, SUPPORTS；Frühwirth-Schnatter, 2006 — nice, SUPPORTS

### J. §3.2 L166–168 — Dirichlet(1) 先验 · should

- CURRENT: `A symmetric Dirichlet(1) prior on **w** imposes a uniform distribution over the simplex, expressing no prior preference among sources.`
- PROPOSED: `A symmetric Dirichlet(1) prior on **w** imposes a uniform distribution over the simplex (Gelman et al., 2013), expressing no prior preference among sources.`
- 引用：Gelman et al., 2013 — should, SUPPORTS（也可在 §5.3 先验敏感性处复用）

### K. §3.3 L176 与 L200–201 — Bootstrap · must / nice

- CURRENT: `draw 100 with-replacement bootstrap resamples of the grains. This yields 12,100`
- PROPOSED: `draw 100 with-replacement bootstrap resamples of the grains (Efron, 1979). This yields 12,100`
- 引用：Efron, 1979 — must, SUPPORTS
- nice（L200–201）：`confirming that ~30 resamples already converge and 100 provides a robustness margin (cf. Efron & Tibshirani, 1993; quantified per b in Table 2)` — Efron & Tibshirani, 1993 — nice, SUPPORTS（B≈50–200 经验法则；⚠ 其 Springer DOI 不解析，见 §3）

### L. §3.4 L189–190 — GPU 实现 · should

- CURRENT: `We therefore re-implemented the sampler as a batched GPU kernel (NVIDIA L20), completing all 181,500 fits in 4475 s (~74 min).`
- PROPOSED: `We therefore re-implemented the sampler as a batched GPU kernel in PyTorch (Ansel et al., 2024; NVIDIA L20), following the established mapping of population-based Monte Carlo onto GPUs (Suchard et al., 2010), completing all 181,500 fits in 4475 s (~74 min).`
- 引用：Ansel et al., 2024 — should, **PARTIAL**（verifier 本地核实 bayes_gpu.py 确为 `import torch` + cuda float64，但该文讲的是 torch.compile，代码没用；只作 PyTorch 2.x 的官方软件引用，同时 Data availability L544 "Software required" 加 `torch (PyTorch)`。49 位作者，是否可截断按 C&G 指南）；Suchard et al., 2010 — should, SUPPORTS（GPU 混合模型 MCMC 最近先例）
- nice：Lee et al., 2010（可与 Suchard 同括号）

### M. §3.4 L202–203 与 Fig. 2 caption L698 — 接受率区间 · 已有文献（修正 inconsistency 1）

- CURRENT: `the acceptance fraction lay within the recommended 0.2–0.5 range for 99.3% of all 181,500 fits`
- PROPOSED: `the acceptance fraction lay within the recommended 0.2–0.5 range (Foreman-Mackey et al., 2013) for 99.3% of all 181,500 fits`
- Fig. 2 L698 `the emcee-recommended healthy band [0.2, 0.5]` → `the emcee-recommended healthy band [0.2, 0.5] (Foreman-Mackey et al., 2013)`
- nice：Roberts et al., 1997 — PARTIAL，仅当改写为 "a band that descends from the random-walk-Metropolis optimal-scaling result of Roberts et al. (1997)" 时可用；不作为 ensemble sampler 的验证。

### N. §3.4 L205 与 Fig. 2 caption L703–704 — 积分自相关时间 · must

- CURRENT: `Per-fit integrated autocorrelation times were not stored for the batched scan; convergence is instead corroborated by the burn-in, the acceptance-fraction diagnostic, and the fit-by-fit agreement with emcee.`
- PROPOSED: `Per-fit integrated autocorrelation times (Hogg & Foreman-Mackey, 2018) were not stored for the batched scan; convergence is instead corroborated by the burn-in, the acceptance-fraction diagnostic, and the fit-by-fit agreement with emcee.`
- 引用：Hogg & Foreman-Mackey, 2018 — must, SUPPORTS（emcee 作者的实用指南，明言自相关时间是最重要的收敛检验；也正好为"未存 IAT"的坦白提供背景）
- nice：Sokal, 1997（τ_int 的标准定义）、Geyer, 1992（MC 标准误/链长依据；注意他反对 burn-in，勿引为 1000 步 burn-in 的背书）、Gelman & Rubin, 1992（PARTIAL，只在改写为 "nor was the multi-sequence R-hat diagnostic (Gelman & Rubin, 1992) computed" 时可用；64 walkers 不是独立链）

### O. §4.2 L266–267 与 L273–274 — 可识别性与 KS · nice

- L273–274 CURRENT: `their Kolmogorov–Smirnov resolvability is ≈ 0.999 and the cosine-normalised`
- PROPOSED: `their resolvability based on the Kolmogorov–Smirnov distance (Vermeesch, 2018; Saylor & Sundell, 2016) is ≈ 0.999 and the cosine-normalised`
- 引用：Vermeesch, 2018 — nice, PARTIAL；Saylor & Sundell, 2016 — nice, PARTIAL（两者只支撑 KS 作为年龄分布比较度量，不支撑 "resolvability" 数值；该指标在 Pan, in preparation 中定义，正文应说明）
- L266–267 nice：Teicher, 1963 — PARTIAL，仅当改写为 `the three-Gaussian mixture is identifiable in the large-n limit (Teicher, 1963); the softness we observe is therefore a finite-grain counting-noise floor, not an intrinsic overlap...`

### P. §4.3 L311–312 — Puetz R-series · must

- CURRENT: `the Puetz R-series in particular yields w_S3 = 0.012–0.016`
- PROPOSED: `the Puetz R-series (Puetz et al., 2024) in particular yields w_S3 = 0.012–0.016`
- 引用：Puetz et al., 2024 — must, **PARTIAL**（数据库本身的正确引用；但作者须确认 dz-global-db 摄入的是 2024 Sci Data 版而非 2021 ESR 版 10.1016/j.earscirev.2021.103745——后者也已 Crossref 核过，可替换）

### Q. §5.2 L364–366 与 L374–376 — 计数统计地板 · must + should

- L364–366 CURRENT: `is set by multinomial counting statistics rather than by any intrinsic inability to separate the 130 and 280 Ma sources.`
- PROPOSED: `is set by multinomial counting statistics (Vermeesch, 2004; Andersen, 2005) rather than by any intrinsic inability to separate the 130 and 280 Ma sources.`
- 引用：Vermeesch, 2004 — must, **PARTIAL**（建立了有限颗粒的二项/多项计数框架与"按目标精度定颗粒数"逻辑，但**没有**推导权重精度的 n^(−1/2) 律，勿引为指数来源）；Andersen, 2005 — should, SUPPORTS（注意 Andersen 结论偏悲观，认为 unmixing 类方法 "of questionable significance"；只引其计数极限，勿引其乐观推论）
- L374–376 CURRENT: `for the great majority of samples the arc:basement uncertainty is *reducible*, and a target grain count to reach any desired precision can be read directly off the n^(−1/2) law.`
- PROPOSED: `for the great majority of samples the arc:basement uncertainty is *reducible*, and a target grain count to reach any desired precision can be read directly off the n^(−1/2) law — grain counts that are now routinely attainable by LA-ICP-MS (Pullen et al., 2014).`
- 引用：Pullen et al., 2014 — should, SUPPORTS（n≈1000 可行，比所需强得多；短从句是新增支撑而非新论断，作者可取舍）

### R. §5.4 L410–411 与 Fig. 6 caption L772 — NMF · must

- CURRENT: `(identical end-members, KL multiplicative updates)`
- PROPOSED: `(identical end-members, KL multiplicative updates; Lee & Seung, 1999)`；Fig. 6 L772 `(identical end-members and KL multiplicative updates; Lee & Seung, 1999)`
- 引用：Lee & Seung, 1999 — must, SUPPORTS（KL 乘法更新的收敛证明在 Lee & Seung 2001 NIPS，无 Crossref DOI，未提出）

### S. §5.5 L459–461 与 L467–468 — 松辽物源与地层 · must（数据源复用）

- L459–461 CURRENT: `consistent with the Songliao Basin drawing on its encircling arc and accretionary terranes rather than the distant craton.`
- PROPOSED: `consistent with the Songliao Basin drawing on its encircling arc and accretionary terranes rather than the distant craton (Song et al., 2022).`
- 引用：Song et al., 2022 — must（同时是 R31370 原始数据源），SUPPORTS（同裂陷期样本仅含显生宙正 εHf 物源）
- L467–468 CURRENT: `It is also the only sample of the SL09xx cluster from the Quantou Formation (the others, Denglouku); we interpret`
- PROPOSED: `It is also the only sample of the SL09xx cluster from the Quantou Formation (the others, Denglouku; Li et al., 2012; Feng et al., 2010); we interpret`
- 引用：Li et al., 2012 — must（SL09xx 原始数据源；其摘要直接支撑 Quantou 以 ~1800 / 2500–2900 Ma 克拉通锆石为主）；Feng et al., 2010 — must（地层单元）
- ⚠ L468–470 "local basement high" 解释：**不建议**贴引用——Li et al. 2012 把 Quantou 克拉通输入归于侵蚀中心向东南迁移到 NCC 基底，Song et al. 2022 把上 Denglouku 的 2.5/1.8 Ga 锆石归于扩大水系的远源搬运，两者都是**替代解释**而非支撑。见 §4。

### T. §5.6 L477–486 — 假设与取样设计 · should + must

- L482–483 后（新增一句，**作者拍板**）：`(iii) Grain counts are modest (n = 31–111). The weights are, moreover, proportions of the zircon population rather than of sediment mass, since zircon fertility and hydraulic sorting bias the detrital record (Malusà et al., 2016).`
- 引用：Malusà et al., 2016 — should, SUPPORTS（"~3 个数量级" fertility 差异出自其摘要；这是省略的第四条假设，物源审稿人会问。因需**新增一句**，不算最小编辑，列出供作者决定）
- L484–486 CURRENT: `future sampling can be designed to a target precision, whereas the two shape-inflated samples`
- PROPOSED: `future sampling can be designed to a target precision (cf. Vermeesch, 2004), whereas the two shape-inflated samples`
- 引用：Vermeesch, 2004 — must（第二处，PARTIAL，措辞用 cf.）

### U. Data and code availability L544 与 L557–561 — 数据源正名 · must（修正 inconsistency 2 & 4）

- L544 CURRENT: `Python ≥ 3.10 with numpy, pandas, pyarrow, scipy, emcee, scikit-learn, and matplotlib; the batched sampler additionally requires a CUDA-capable GPU.`
- PROPOSED: `Python ≥ 3.10 with numpy, pandas, pyarrow, scipy, emcee, scikit-learn, and matplotlib; the batched sampler additionally requires torch (PyTorch; Ansel et al., 2024) and a CUDA-capable GPU.`
- L557–561 CURRENT: `The input samples are compiled from published open datasets: NCC north-margin samples from doi:10.1016/j.epsl.2006.04.029 and doi:10.1080/00206814.2016.1168323; Songliao samples from the OneDZ compilation (doi:10.1016/j.gr.2012.04.002) and the Puetz compilation (doi:10.1016/j.sedgeo.2022.106133; doi:10.1016/j.gsf.2022.101377).`
- PROPOSED: `The input samples are compiled from published open datasets: NCC north-margin samples from Yang et al. (2006) and Xu et al. (2016); Songliao samples SL0929–SL0932 from Li et al. (2012), accessed via the OneDZ database (Li et al., 2026), and the R31354 and R31370 series from Wang et al. (2022) and Song et al. (2022), respectively, accessed via the Puetz et al. (2024) database.`
- 引用：Yang et al., 2006 — must, **PARTIAL**（05FW007/008 归属仅经 OneDZ 属性间接确认；Crossref 作者字段是退化的单缩写 `Yang, J., Wu, F., ...`，印刷署名几乎肯定是 `Yang, J.-H., Wu, F.-Y., Shao, J.-A., Wilde, S.A., Xie, L.-W., Liu, X.-M.`，作者须对 PDF 定稿）；Xu et al., 2016 — must, SUPPORTS；Li et al., 2012 — must, SUPPORTS；Wang et al., 2022 — must, SUPPORTS；Song et al., 2022 — must, SUPPORTS；Li et al., 2026 — must, SUPPORTS（OneDZ 数据论文；作者须确认查询的是该版本，其 Zenodo 为 10.5281/zenodo.19690702）；Puetz et al., 2024 — must, PARTIAL（同 P）

### V. Table 1 notes L657–658 与 Fig. 1 caption L691–692 — 坐标来源 · must

- CURRENT（两处近同）: `compiled from the dz-global-db compilation (OneDZ/Puetz sources)` / `(compiled from the dz-global-db compilation, OneDZ/Puetz sources)`
- PROPOSED: `compiled via the OneDZ (Li et al., 2026) and Puetz et al. (2024) databases` / `(compiled via the OneDZ, Li et al., 2026, and Puetz et al., 2024, databases)`
- 说明："dz-global-db" 无任何可引条目（见 §4）；若保留该名，须给 Zenodo/ESSD 记录。

### 采纳统计

| tier | 条数 | 条目 |
|---|---|---|
| must | 16 | Gehrels 2014; Tipton 2022; Mosegaard & Tarantola 1995; Sambridge & Mosegaard 2002; Feng 2010; Efron 1979; Hogg & Foreman-Mackey 2018; Puetz 2024; Vermeesch 2004; Lee & Seung 1999; Song 2022; Li 2012; Yang 2006; Xu 2016; Wang S. 2022; Li K. 2026 |
| should | 9 | Wang P.-J. 2016; Xiao 2003; Zhao 2005; Gelman 2013; Ansel 2024; Suchard 2010; Andersen 2005; Pullen 2014; Malusà 2016 |
| nice | 19 | Redner & Walker 1984; Jasra 2006; Tarantola 2005; Xu W.-L. 2013; Wu 2011; Windley 2007; Wilde 2015; Aitchison 1986; McLachlan & Peel 2000; Frühwirth-Schnatter 2006; Efron & Tibshirani 1993; Lee A. 2010; Roberts 1997; Sokal 1997; Geyer 1992; Gelman & Rubin 1992; Vermeesch 2018; Saylor & Sundell 2016; Teicher 1963 |

## 3. 新 References 条目（字母序，稿件格式）

> 合并到现有列表时注意：`Li, K.` 2026 排在现有 `Li, H.` 2021 之后、`Li, S.-Q.` 2012 之前；`Xiao, W., Santosh, M., 2014`（两作者）按 Elsevier 惯例排在 `Xiao, W., Windley, B.F., ...` 2003（多作者）之前；`Zhao, G., Sun, M., ...` 2005 排在 `Zhao, G., 2001` 与 `Zhao, G., Cawood, ...` 2012 之间；`Wu, F.-Y.` 2011 在现有 `Wu, J.T.-J.` 2022 之前。Crossref 为 Title Case 的期刊标题已按稿件 sentence-case 惯例转换；书名保留 Title Case。

- Aitchison, J., 1986. The Statistical Analysis of Compositional Data. Chapman & Hall, London. <!-- nice; Springer 版 DOI 10.1007/978-94-009-4109-0 在 Crossref 有记录但 2026-09-02 落地页 404，故按 verifier 建议引 Chapman & Hall 版不带 DOI；若投稿时 DOI 恢复可改 "Springer Netherlands, Dordrecht. doi:10.1007/978-94-009-4109-0" -->
- Andersen, T., 2005. Detrital zircons as tracers of sedimentary provenance: limiting conditions from statistics and numerical simulation. Chemical Geology 216 (3–4), 249–270. doi:10.1016/j.chemgeo.2004.11.013 <!-- should -->
- Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., Hirsh, B., Huang, S., Kalambarkar, K., Kirsch, L., Lazos, M., Lezcano, M., Liang, Y., Liang, J., Lu, Y., Luk, C.K., Maher, B., Pan, Y., Puhrsch, C., Reso, M., Saroufim, M., Siraichi, M.Y., Suk, H., Zhang, S., Suo, M., Tillet, P., Zhao, X., Wang, E., Zhou, K., Zou, R., Wang, X., Mathews, A., Wen, W., Chanan, G., Wu, P., Chintala, S., 2024. PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph compilation. In: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24). ACM, pp. 929–947. doi:10.1145/3620665.3640366 <!-- should; PARTIAL; 49 作者是否截断按 C&G 指南 -->
- Efron, B., 1979. Bootstrap methods: another look at the jackknife. The Annals of Statistics 7 (1), 1–26. doi:10.1214/aos/1176344552 <!-- must -->
- Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap. Chapman & Hall, New York. <!-- nice; Crossref DOI 10.1007/978-1-4899-4541-9 已注册但不解析(404)，故不带 DOI；可解析的替代为 CRC 1994 电子版 doi:10.1201/9780429246593，但年份须同步改 1994 -->
- Feng, Z.-Q., Jia, C.-Z., Xie, X.-N., Zhang, S., Feng, Z.-H., Cross, T.A., 2010. Tectonostratigraphic units and stratigraphic sequences of the nonmarine Songliao basin, northeast China. Basin Research 22 (1), 79–95. doi:10.1111/j.1365-2117.2009.00445.x <!-- must; Crossref 中文作者姓名字段互换，此处为正确的姓前读法，勿照抄 Crossref family 字段 -->
- Frühwirth-Schnatter, S., 2006. Finite Mixture and Markov Switching Models. Springer Series in Statistics. Springer, New York. doi:10.1007/978-0-387-35768-3 <!-- nice; Crossref 书记录无作者字段，作者身份经 OpenAlex + S2 双确认 -->
- Gehrels, G., 2014. Detrital zircon U-Pb geochronology applied to tectonics. Annual Review of Earth and Planetary Sciences 42 (1), 127–149. doi:10.1146/annurev-earth-050212-124012 <!-- must -->
- Gelman, A., Rubin, D.B., 1992. Inference from iterative simulation using multiple sequences. Statistical Science 7 (4), 457–472. doi:10.1214/ss/1177011136 <!-- nice; PARTIAL，仅配合改写使用 -->
- Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., Rubin, D.B., 2013. Bayesian Data Analysis, third ed. Chapman and Hall/CRC. doi:10.1201/b16018 <!-- should; Crossref 无出版地，如期刊要求可补 Boca Raton -->
- Geyer, C.J., 1992. Practical Markov chain Monte Carlo. Statistical Science 7 (4), 473–483. doi:10.1214/ss/1177011137 <!-- nice -->
- Hogg, D.W., Foreman-Mackey, D., 2018. Data analysis recipes: using Markov chain Monte Carlo. The Astrophysical Journal Supplement Series 236 (1), 11. doi:10.3847/1538-4365/aab76e <!-- must -->
- Jasra, A., Stephens, D.A., Gallagher, K., Holmes, C.C., 2006. Bayesian mixture modelling in geochronology via Markov chain Monte Carlo. Mathematical Geology 38 (3), 269–300. doi:10.1007/s11004-005-9019-3 <!-- nice -->
- Lee, A., Yau, C., Giles, M.B., Doucet, A., Holmes, C.C., 2010. On the utility of graphics cards to perform massively parallel simulation of advanced Monte Carlo methods. Journal of Computational and Graphical Statistics 19 (4), 769–789. doi:10.1198/jcgs.2010.10039 <!-- nice -->
- Lee, D.D., Seung, H.S., 1999. Learning the parts of objects by non-negative matrix factorization. Nature 401 (6755), 788–791. doi:10.1038/44565 <!-- must -->
- Li, K., Hu, X., Chai, R., Yang, J., Xue, W., Pan, Y., Li, T., Fang, C., Ma, A., Huang, H., Guo, Q., Yang, W., Hu, L., Qi, L., Chen, G., Sun, G., Zhang, S., Deng, T., Li, K., Sun, J., Gao, B., 2026. OneDZ: a global detrital zircon database and implications for constructing giant geoscience database. Earth System Science Data 18 (5), 3671–3696. doi:10.5194/essd-18-3671-2026 <!-- must -->
- Li, S.-Q., Chen, F., Siebel, W., Wu, J.-D., Zhu, X.-Y., Shan, X.-L., Sun, X.-M., 2012. Late Mesozoic tectonic evolution of the Songliao basin, NE China: evidence from detrital zircon ages and Sr–Nd isotopes. Gondwana Research 22 (3–4), 943–955. doi:10.1016/j.gr.2012.04.002 <!-- must -->
- Malusà, M.G., Resentini, A., Garzanti, E., 2016. Hydraulic sorting and mineral fertility bias in detrital geochronology. Gondwana Research 31, 1–19. doi:10.1016/j.gr.2015.09.002 <!-- should; 需新增一句，作者拍板 -->
- McLachlan, G., Peel, D., 2000. Finite Mixture Models. Wiley Series in Probability and Statistics. Wiley, New York. doi:10.1002/0471721182 <!-- nice -->
- Mosegaard, K., Tarantola, A., 1995. Monte Carlo sampling of solutions to inverse problems. Journal of Geophysical Research: Solid Earth 100 (B7), 12431–12447. doi:10.1029/94JB03097 <!-- must -->
- Puetz, S.J., Spencer, C.J., Condie, K.C., Roberts, N.M.W., 2024. Enhanced U-Pb detrital zircon, Lu-Hf zircon, δ18O zircon, and Sm-Nd whole rock global databases. Scientific Data 11 (1), 56. doi:10.1038/s41597-023-02902-9 <!-- must; PARTIAL，作者须确认摄入的是此版本 -->
- Pullen, A., Ibáñez-Mejía, M., Gehrels, G.E., Ibáñez-Mejía, J.C., Pecha, M., 2014. What happens when n = 1000? Creating large-n geochronological datasets with LA-ICP-MS for geologic investigations. Journal of Analytical Atomic Spectrometry 29 (6), 971–980. doi:10.1039/c4ja00024b <!-- should -->
- Redner, R.A., Walker, H.F., 1984. Mixture densities, maximum likelihood and the EM algorithm. SIAM Review 26 (2), 195–239. doi:10.1137/1026034 <!-- nice; PARTIAL -->
- Roberts, G.O., Gelman, A., Gilks, W.R., 1997. Weak convergence and optimal scaling of random walk Metropolis algorithms. The Annals of Applied Probability 7 (1), 110–120. doi:10.1214/aoap/1034625254 <!-- nice; PARTIAL -->
- Sambridge, M., Mosegaard, K., 2002. Monte Carlo methods in geophysical inverse problems. Reviews of Geophysics 40 (3). doi:10.1029/2000RG000089 <!-- must; Crossref/OpenAlex/S2 均无页码或文献号，常见的 "1009" 需作者核 PDF 后补 -->
- Saylor, J.E., Sundell, K.E., 2016. Quantifying comparison of large detrital geochronology data sets. Geosphere 12 (1), 203–220. doi:10.1130/ges01237.1 <!-- nice; PARTIAL -->
- Sokal, A., 1997. Monte Carlo methods in statistical mechanics: foundations and new algorithms. In: DeWitt-Morette, C., Cartier, P., Folacci, A. (Eds.), Functional Integration: Basics and Applications, NATO ASI Series B, vol. 361. Springer US, Boston, MA, pp. 131–192. doi:10.1007/978-1-4899-0319-8_6 <!-- nice -->
- Song, Y., Ren, J., Liu, K., Lyu, D., Feng, X., Liu, Y., Stepashko, A., 2022. Syn-rift to post-rift tectonic transition and drainage reorganization in continental rifting basins: detrital zircon analysis from the Songliao Basin, NE China. Geoscience Frontiers 13 (3), 101377. doi:10.1016/j.gsf.2022.101377 <!-- must -->
- Suchard, M.A., Wang, Q., Chan, C., Frelinger, J., Cron, A., West, M., 2010. Understanding GPU programming for statistical computation: studies in massively parallel massive mixtures. Journal of Computational and Graphical Statistics 19 (2), 419–438. doi:10.1198/jcgs.2010.10016 <!-- should -->
- Tarantola, A., 2005. Inverse Problem Theory and Methods for Model Parameter Estimation. Society for Industrial and Applied Mathematics. doi:10.1137/1.9780898717921 <!-- nice; PARTIAL -->
- Teicher, H., 1963. Identifiability of finite mixtures. The Annals of Mathematical Statistics 34 (4), 1265–1269. doi:10.1214/aoms/1177703862 <!-- nice; PARTIAL -->
- Tipton, J.R., Sharman, G.R., Johnstone, S.A., 2022. A Bayesian nonparametric approach to unmixing detrital geochronologic data. Mathematical Geosciences 54 (1), 151–176. doi:10.1007/s11004-021-09961-x <!-- must; PARTIAL（作为先例而非支撑） -->
- Vermeesch, P., 2004. How many grains are needed for a provenance study? Earth and Planetary Science Letters 224 (3–4), 441–451. doi:10.1016/j.epsl.2004.05.037 <!-- must; PARTIAL -->
- Vermeesch, P., 2018. Dissimilarity measures in detrital geochronology. Earth-Science Reviews 178, 310–321. doi:10.1016/j.earscirev.2017.11.027 <!-- nice; PARTIAL -->
- Wang, P.-J., Mattern, F., Didenko, N.A., Zhu, D.-F., Singer, B., Sun, X.-M., 2016. Tectonics and cycle system of the Cretaceous Songliao Basin: an inverted active continental margin basin. Earth-Science Reviews 159, 82–102. doi:10.1016/j.earscirev.2016.05.004 <!-- should; PARTIAL -->
- Wang, S., Cheng, Y., Jin, R., Miao, P., Zhang, T., Xu, Z., Ao, C., Teng, X., Cheng, X., 2022. Detrital zircon U-Pb ages of the Cretaceous strata in the southern Songliao Basin, NE China: constraints on basin-and-range evolution. Sedimentary Geology 433, 106133. doi:10.1016/j.sedgeo.2022.106133 <!-- must -->
- Wilde, S.A., 2015. Final amalgamation of the Central Asian Orogenic Belt in NE China: Paleo-Asian Ocean closure versus Paleo-Pacific plate subduction — a review of the evidence. Tectonophysics 662, 345–362. doi:10.1016/j.tecto.2015.05.006 <!-- nice -->
- Windley, B.F., Alexeiev, D., Xiao, W., Kröner, A., Badarch, G., 2007. Tectonic models for accretion of the Central Asian Orogenic Belt. Journal of the Geological Society 164 (1), 31–47. doi:10.1144/0016-76492006-022 <!-- nice -->
- Wu, F.-Y., Sun, D.-Y., Ge, W.-C., Zhang, Y.-B., Grant, M.L., Wilde, S.A., Jahn, B.-M., 2011. Geochronology of the Phanerozoic granitoids in northeastern China. Journal of Asian Earth Sciences 41 (1), 1–30. doi:10.1016/j.jseaes.2010.11.014 <!-- nice; PARTIAL -->
- Xiao, W., Windley, B.F., Hao, J., Zhai, M., 2003. Accretion leading to collision and the Permian Solonker suture, Inner Mongolia, China: termination of the central Asian orogenic belt. Tectonics 22 (6), 2002TC001484. doi:10.1029/2002TC001484 <!-- should; Crossref 无页码，用 article-number；印刷引用号 1069 需核 PDF -->
- Xu, H., Liu, Y., Kuang, H., Peng, N., 2016. Sedimentary response to the intracontinental orogenic process: insight from the anatomy of a small Mesozoic basin in western Yanshan, northern North China. International Geology Review 58 (12), 1528–1556. doi:10.1080/00206814.2016.1168323 <!-- must -->
- Xu, W.-L., Pei, F.-P., Wang, F., Meng, E., Ji, W.-Q., Yang, D.-B., Wang, W., 2013. Spatial–temporal relationships of Mesozoic volcanic rocks in NE China: constraints on tectonic overprinting and transformations between multiple tectonic regimes. Journal of Asian Earth Sciences 74, 167–193. doi:10.1016/j.jseaes.2013.04.003 <!-- nice -->
- Yang, J., Wu, F., Shao, J., Wilde, S., Xie, L., Liu, X., 2006. Constraints on the timing of uplift of the Yanshan Fold and Thrust Belt, North China. Earth and Planetary Science Letters 246 (3–4), 336–352. doi:10.1016/j.epsl.2006.04.029 <!-- must; PARTIAL; 缩写按 Crossref 退化记录，印刷署名几乎肯定为 Yang, J.-H., Wu, F.-Y., Shao, J.-A., Wilde, S.A., Xie, L.-W., Liu, X.-M.，投稿前核 PDF -->
- Zhao, G., Sun, M., Wilde, S.A., Li, S., 2005. Late Archean to Paleoproterozoic evolution of the North China Craton: key issues revisited. Precambrian Research 136 (2), 177–202. doi:10.1016/j.precamres.2004.10.002 <!-- should; Crossref 第四作者 given/family 互换，正确为 Li, S. -->

## 4. 仍无引用支撑的句子

按严重程度排序；前三条是**事实层**问题，不是补一条文献能解决的。

1. **§2 L106 / Abstract / Table 1 / Fig. 1 — "fifteen Lower Cretaceous (K1) samples" 对 NCC 四样本存疑。** verifier 从 Zenodo 包 CSV header 读到：05FW007 = Longmen Fm、05FW008 = Jiulongshan Fm（源 Yang et al. 2006）、100821-1/-3 = Tuchengzi Fm（源 Xu et al. 2016）。Xu et al. 2016 摘要把 Xiahuayuan/Longmen 定为早–中侏罗世、Tuchengzi 为晚侏罗–早白垩世；Jiulongshan 通常为中侏罗世。OneDZ 给 05FW007 的沉积年龄 131.5 Ma 与之矛盾。**作者须对原始论文逐样本核定层位与年龄**，并在 §2 或 Table 1 加一列 Formation；若确为侏罗系，全稿 "K1" 措辞、标题与 Highlights 都要改。这是 data-vs-claim 级别的问题。
2. **Table 1 L636–637 — 05FW007/008 坐标 (40.833°N, 115.833°E)。** 本地 curation 文件标记为 `basin_from_box`（包围盒质心），而 Yang et al. 2006 的 Xishan（北京西山）剖面约在 39.9–40.0°N、115.9–116.1°E。加引用时顺手从原文核正坐标，Fig. 1 同步。
3. **§5.5 L468–470 / Conclusion 6 — "proximal sourcing from a local basement high of exposed North China cratonic basement"。** 无任何文献支撑松辽东南部存在出露的 NCC 基底高地；两篇原始数据源给出的是**替代解释**（Li et al. 2012：侵蚀中心向东南迁移到 NCC 基底；Song et al. 2022：扩大水系远源搬运克拉通碎屑，R31370-S12653 w_S3 = 0.59 可能就是上 Denglouku 样本）。建议要么找松辽盆地基底地质/前寒武系出露的文献，要么改写为 "proximal or drainage-delivered cratonic sourcing (cf. Li et al., 2012; Song et al., 2022)"——后者是新论断改动，作者拍板。
4. **§1 L75 — Sundell & Saylor 2017 ≠ bootstrap CI。** 已在 §2-B 改写；剩余待办是核 Sharman & Johnstone 2017 方法节是否用 bootstrap。
5. **Table 1 notes / Fig. 1 caption — "dz-global-db compilation"。** 无可引条目。DataCite 查到作者自己的 Zenodo 10.5281/zenodo.20728422（ESSD 配套 East China + Korea 汇编，2026-06-17）可能是它，也可能不是（用户笔记把两者当不同项目）。要么给正确的 Zenodo/ESSD 记录，要么按 §2-V 删掉该名只留 OneDZ/Puetz。
6. **Fig. 1 caption L686 — "Songliao wide-rift basin"。** "wide rift" 是特定构造分类，无来源；Wang et al. 2016 反而把它归为 inverted active continental margin basin。finder 提到 Meng 2003 (Tectonophysics 369, doi:10.1016/S0040-1951(03)00195-1) 但**本轮未经 verifier 核验**，不列入建议；作者自行核后再用，或改为中性的 "Songliao Basin"。
7. **Fig. 1 caption L688–689 — Tanlu 左行走滑、Mongol–Okhotsk 缝合带位置。** 只挂 "after Pan (submitted)"。需一篇 Tan-Lu 断裂运动学原始文献（Zhu G. 等系列）与一篇 Mongol–Okhotsk 综述；本轮未核。
8. **Fig. 1 caption L692–693 — FGDC TM 11A02、Okabe–Ito 配色。** 灰色文献无 DOI：FGDC-STD-013-2006 *Digital Cartographic Standard for Geologic Map Symbolization*；Okabe & Ito 2008 网页资源。应按 Elsevier 灰色文献格式（机构、年份、URL、访问日期）列入 References。
9. **§5.6 L481 — "sparsity-promoting Dirichlet(α < 1)"。** finder 提到 Rousseau & Mengersen 2011 (JRSS-B 73(5), doi:10.1111/j.1467-9868.2011.00781.x) 但本轮未核；可不引，若引须先核。
10. **§5.3 L395–397 — "a standard objection to any prior-based unmixing — that the answer is a restatement of the prior"。** Berger 1994 因作者元数据问题被否（见 §5）；Stock et al. 2018 (PeerJ 6, e5096) 由 finder 提出但未核。可保持不引（是常识性表述），或核 Stock 2018 后引为贝叶斯示踪混合模型中该质疑的标准出处。
11. **§3.4 L203 — 0.2–0.5 接受率区间的原始出处** Gelman, Roberts & Gilks 1996（Bayesian Statistics 5, OUP）无 Crossref DOI；引 Foreman-Mackey et al. 2013 已足够（§2-M）。
12. **§3.4 — PyTorch 原始论文** Paszke et al. 2019 (NeurIPS 32) 无 Crossref DOI；已用 Ansel et al. 2024 替代（§2-L）。
13. **§5.4 / Fig. 6 — KL 乘法更新的收敛证明** Lee & Seung 2001 (NIPS 13) 无 Crossref DOI；已引 1999 Nature（§2-R）。
14. **§4.2 L281–283 / §5.3 / Fig. 4 — law-of-total-variance split。** 教科书恒等式，无需引用；若编辑坚持，Gelman et al. 2013 可兼用。
15. **§3.4 L195–197 — emcee re-seeding scatter ~0.19。** 内部经验结果，无外部引用适用；审稿人可能要求用 CPU emcee 参考跑的 τ_int/ESS 佐证。
16. **§2 L121 — "provide a broad late-Palaeozoic source"。** 现引 Liu 2017 / Xiao & Santosh 2014 可用；注意 Wu et al. 2011 的头条结论是古生代花岗岩分布"没有以前认为的那么广"，不要把它加到这一分句。

## 5. 落选记录

- **Sambridge & Compston 1994** (doi:10.1016/0012-821x(94)90157-0) — 落选原因是机械性的：候选缩写 `Sambridge, M.` 与 Crossref `M.S. Sambridge` 不符；内容 SUPPORTS（锆石年龄有限混合模型的经典源头，适合 §3.1）。verifier 建议以 `Sambridge, M.S., Compston, W., 1994. Mixture modeling of multi-component data sets with application to ion-probe zircon ages. Earth and Planetary Science Letters 128 (3–4), 373–390. doi:10.1016/0012-821X(94)90157-0` 重新接纳；本文按规则不列入建议，作者可自行决定。
- **Saylor, Sundell & Sharman 2019** (doi:10.1016/j.epsl.2019.01.044) — 同为缩写不符（候选漏了第二缩写 J.E./K.E./G.R.）；内容 SUPPORTS（NMF 系 unmixing 方法，适合 §1 L69 方法列表与 §5.4 companion 方法谱系）。修正后条目 `Saylor, J.E., Sundell, K.E., Sharman, G.R., 2019. Characterizing sediment sources by non-negative matrix factorization of detrital geochronological data. Earth and Planetary Science Letters 512, 46–58. doi:10.1016/j.epsl.2019.01.044` 可用，同上由作者决定。
- **Berger 1994** (doi:10.1007/BF02562676) — 内容 SUPPORTS（§5.3 稳健贝叶斯/先验敏感性框架），但该 DOI 覆盖含讨论与答复的整篇（Test 3(1), 5–124），Crossref 列 23 位作者；候选按单作者引与实时记录不符。若要用须自觉选定一种写法（`Berger, J.O., 1994. An overview of robust Bayesian analysis (with discussion). Test 3 (1), 5–124.` 或 23 作者全列），本文不代作决定。