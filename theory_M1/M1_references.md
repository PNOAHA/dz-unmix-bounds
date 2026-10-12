# M1 理论线 · §3 可识别性定理 — 注释书目与决策记录

> **建立日期**：2026-06-11
> **所属**：`dz-unmix-bounds` 第二篇方法论文（适用边界）的 §3 理论支柱
> **状态**：文献已拉齐、框架已钉死、§3 命题骨架已成形；尚未动笔证明
> **配套文件**：[`M1_references.bib`](M1_references.bib)（BibTeX，可直接 `\cite`）

---

## 0. 这份文件是什么 / 怎么来的

这是把 bounds 论文的经验失败相图（K×ρ）升级成**解析可识别性定理**（代号 M1）的文献底座与下笔起点。

**溯源链（2026-06-10 ~ 06-11）**：
1. 一轮 4 视角发散 + 对抗综合，产出"与数学的融合"战略路线图，M1 被判为最强机会。
2. 一轮 7-agent 对抗验证，发现 M1 有概念裂缝（误把 supervised 当 blind），需先钉范式。
3. **读 bounds 自身代码定死范式**：`src/adapter_cg.py:186` + `dz-unmix-cg/dz_unmixing_experiment.py:206-240` 证明——**端元矩阵 H 在推断时已知，只反演权重 W，算法是 KL 散度乘法更新的固定字典 NMF（Lee-Seung 型），不是 NNLS、不是 Frobenius、不是 blind NMF**。
4. 一轮 5-簇并行文献检索（46 条，带核验状态），组装成本文件。

---

## 1. 框架复述：该引什么、丢什么

**M1 数学内核** = 已知端元密度 H、概率单纯形约束（w≥0, Σw=1）、KL 散度（=多项式/Poisson 极大似然）准则下反演混合权重 w。

| | 内容 |
|---|---|
| **该引** | (A) 有限混合"线性无关 ⟺ 可识别"母定理（Yakowitz-Spragins / Teicher）<br>(B) 已知字典 Poisson-ML 乘法更新收敛（Shepp-Vardi / Lee-Seung 2001）<br>(C) GLM Fisher 信息 = HᵀWH 局部曲率（McCullagh-Nelder）<br>(D) 高斯重叠积分 + Gram→条件数闭式（Matrix Cookbook / Horn-Johnson） |
| **该丢** | separable / blind NMF 唯一性整支（Donoho-Stodden、Arora-Ge SPA、Gillis-Vavasis σ_min；地学侧 Saylor 2019 blind NMF、Graham 2025 张量分解）——**它们解 H 未知问题，与 M1 的 known-H 不是同一命题** |

---

## 2. 按定理结构分组的核心文献

> 状态标注：✅ verified-exists（检索确认存在） · ⚠️ likely / could-not-verify（**投稿前须潘一手核**）

### (A) 可识别性地基 — 钉死"端元线性无关 ⟺ 权重可识别"

- ✅ **Yakowitz & Spragins 1968.** On the Identifiability of Finite Mixtures. *Ann. Math. Stat.* 39(1):209-214. DOI `10.1214/aoms/1177698520` — **母定理，最直接可引（充要条件版）**
- ✅ Teicher 1963. Identifiability of Finite Mixtures. *Ann. Math. Stat.* 34(4):1265-1269. DOI `10.1214/aoms/1177703862`
- ✅ Titterington, Smith & Makov 1985. *Statistical Analysis of Finite Mixture Distributions.* Wiley. ISBN 9780471907633 — 教科书出处，第3章
- ✅ Ramaswamy, Scott & Tewari 2016. Mixture Proportion Estimation via Kernel Embedding. *ICML*; arXiv:1603.02501 — 现代 MPE；注意是 1-成分 vs 混合设定，需说明与 K-端元单纯形差异

### (B) 条件数 / Fisher 信息 — 把 ρ（共线性）和 N（样本量）连进一个公式

- ✅ **McCullagh & Nelder 1989.** *Generalized Linear Models* 2nd. Chapman & Hall/CRC. ISBN 9780412317606 — **Fisher = XᵀWX, W=diag(1/μ) 的理论基石**；固定 H 时 H 当设计阵，μ=Hw
- ✅ Higham 2002. *Accuracy and Stability of Numerical Algorithms* 2nd. SIAM. DOI `10.1137/1.9780898718027` — 约束 LS 扰动界 ∝ κ
- ✅ Golub & Van Loan 2013. *Matrix Computations* 4th. JHU Press. ISBN 9781421407944 — κ(AᵀA)=κ(A)²
- ✅ Shepp & Vardi 1982. ML Reconstruction for Emission Tomography. *IEEE TMI* 1(2):113-122. DOI `10.1109/TMI.1982.4307558` — **固定字典 + Poisson 似然乘法 EM 严格收敛（=Richardson-Lucy/EML-EM），与 known-H KL 更新同型**
- ✅ Carreira-Perpiñán & Renals 2000. Practical Identifiability of Finite Mixtures of Multivariate Bernoulli. *Neural Computation* 12(1):141-152. DOI `10.1162/089976600300015925` — "理论可识别 ≠ 实用可识别"实证先例，支撑相图连续/分级命题
- ✅ Ho & Nguyen 2016/2019. Singularity structures … in finite mixtures. arXiv:1609.02655（→ *SIAM J. Math. Data Sci.* 2019）— 退化⇒估计速率恶化模板。**注意其框架近 blind，只借精神不套速率结论**
- ✅ Cemgil 2009. Bayesian Inference for NMF Models. *Comput. Intell. Neurosci.* 2009:785152. DOI `10.1155/2009/785152` — KL-NMF ↔ Poisson 生成模型，给有限样本噪声触发病态的概率框架
- ⚠️ **Hall & Titterington 1985（及 Hall 1981）非参数混合比例估计** — **could-not-verify**。"成分重叠↑ ⇒ 渐近方差膨胀"的统计学祖本，正中目标公式，**但确切卷期页只在二手转述中见到，投稿前必须调 JRSS-B（及 Statist. Probab. Lett.）原文一手核**。未核前不要当确定引文写进定理。
- ⚠️ Hsieh & Dhillon 2011. Fast Coordinate Descent for NMF. *KDD*:1064-1072. DOI `10.1145/2020408.2020577` — **likely-exists**。KL-NMF Hessian 含 V/(WH)² 曲率项，是 GLM→KL-NMF 不动点曲率桥梁

### (C) KL-NMF 算法层 — 钉死"算法是 KL 乘法更新，非 NNLS/Frobenius/blind"

- ✅ **Lee & Seung 2001.** Algorithms for Non-negative Matrix Factorization. *NIPS* 13:556-562 — **KL 乘法更新 + auxiliary-function 单调收敛的精确出处，核心引文**
- ✅ Lee & Seung 1999. Learning the parts of objects by NMF. *Nature* 401:788-791. DOI `10.1038/44565` — NMF 奠基/背景
- ✅ Févotte & Idier 2011. Algorithms for NMF with the β-Divergence. *Neural Computation* 23(9):2421-2456. DOI `10.1162/NECO_a_00168` — β=1 即 KL=Poisson 噪声模型
- ✅ Cichocki, Zdunek, Phan & Amari 2009. *Nonnegative Matrix and Tensor Factorizations.* Wiley. DOI `10.1002/9780470747278` — "KL=多项式/Poisson ML"最规范出处

> multinomial（固定每样本 N 颗）vs Poisson（条件化）的差别一句话交代即可（标准结果），**不算 gap**。

### (D) 地学 EMMA 先例与差异化 — 锚回 DZ 谱系 + 标出已占领地

- ✅ **Sharman & Johnstone 2017.** Sediment unmixing using detrital geochronology. *EPSL.* DOI `10.1016/j.epsl.2017.07.044` — **top-down = known-H 的直接地学先例，最该正面区分（指出非唯一性但无 ρ→条件数判据）**
- ✅ Sundell & Saylor 2017. Unmixing DZ age distributions (DZmix). *G-Cubed* 18(8):2872-2886. DOI `10.1002/2016GC006774` — known-H 加权反演（蒙特卡洛路线）
- ✅ Tipton, Sharman & Johnstone 2022. A Bayesian Nonparametric Approach to Unmixing DZ Data. *Math. Geosci.* DOI `10.1007/s11004-021-09961-x` — 同刊；**佐证目标刊收这类方法论文**
- ✅ Weltje 1997. End-member modeling of compositional data. *Math. Geol.* 29(4):503-549. DOI `10.1007/BF02775085` — EMMA 奠基（启发式唯一性，无充分条件）
- ✅ Paterson & Heslop 2015. New methods for unmixing sediment grain size data (AnalySize). *G-Cubed* 16(12):4494-4506. DOI `10.1002/2015GC006070` — 主流 EMMA 工具，作"现有工具皆 blind/启发式、无解析病态判据"对照
- ✅ Dietze & Dietze 2019. Grain-size unmixing with EMMAgeo. *EGQSJ* 68:29-46. DOI `10.5194/egqsj-68-29-2019`
- ⚠️ **Dietze & Dietze 2022. Application of EMMA to grain-size data: Constraints and limitations. *Sedimentology* 69(5). DOI `10.1111/sed.12929`** — verified-exists 但**正文付费墙未取到**。地学侧最接近"失效相图"主题的经验论文。**必须读全文确认它没偷偷给出 KS 或条件数阈值**，否则差异化叙事有风险。
- ⚠️ **Graham, Richardson, Friedlander & Saylor 2025. Tracing Sedimentary Origins via Constrained Tensor Factorization. *Math. Geosci.* 57:601-628. DOI `10.1007/s11004-024-10175-0`** — verified-exists 但**作者全名/页码未一手确认**。**目标投稿圈最强竞品**（同刊，应用数学家+地质学家联手）。blind 张量分解，必须正面区分。
- ⚠️ **Saylor, Jordan & Sundell 2019. Characterizing sediment sources by NMF of DZ data. *EPSL* 512:46-58. DOI `10.1016/j.epsl.2019.01.044`** — verified-exists 但**PDF 损坏、识别性段落未读到**。blind NMF（M1 要丢弃谱系在地学的落地），作"地学已用 NMF 但都是 blind"反衬。
- ✅ **Qi & van der Heijden 2025.** A review of NMF, PLSA, LBA, EMA, LCA with focus on identifiability. arXiv:2512.22282 — Theorem 5 证 EMA 唯一 iff NMF 唯一，**但只处理 blind**、通用统计综述、不碰 DZ/KS/条件数。**M1 的 known-H + KS→Gram 没被它占。** ⚠️ 潘原备忘记成"Qi/Chen/van der Heijden"——**无 Chen，应删**。

### (E) 桥接引理：KS → 几何 — 两端现成砖 + 中间缺口

- ✅ **Horn & Johnson 2013.** *Matrix Analysis* 2nd. Cambridge UP. ISBN 9780521548236 — **Gram 正定⟺线性无关；2×2 Gram 本征值 1±cosθ、κ=(1+|cosθ|)/(1−|cosθ|)；Geršgorin 给 λ_min(HHᵀ) 下界——桥的代数骨架**
- ✅ **Petersen & Pedersen 2012.** *The Matrix Cookbook* §8.1.8（高斯密度乘积）. TU Denmark（在线，无 DOI）— **高斯重叠积分闭式 ⟨h_i,h_j⟩∝exp(−Δμ²/2(σ²ᵢ+σ²ⱼ))——"间距→Gram"捷径核心闭式**
- ✅ Gibbs & Su 2002. On Choosing and Bounding Probability Metrics. *Int. Stat. Rev.* 70(3):419-435（arXiv:math/0209021）. DOI `10.1111/j.1751-5823.2002.tb00178.x` — **关键：KS 只能被别的度量上界，反方向无通用 arrow**
- ✅ Gaunt & Li 2022. Bounding Kolmogorov distances through Wasserstein. arXiv:2201.12087（→ *JMAA*）— d_KS ≤ C·(M·d_p)^{p/(p+1)}；**方向与桥接所需相反，只能作 caveat**
- ✅ Sason & Verdú 2016. f-Divergence Inequalities (incl. reverse Pinsker). *IEEE Trans. Inf. Theory* 62(11):5973-6006. DOI `10.1109/TIT.2016.2603151` — reverse 方向需密度下界/有界 likelihood ratio，支撑"通用反向界不存在、必须靠高斯特解"
- ✅ van der Meer & Jia 2012. Collinearity and orthogonality of endmembers in linear spectral unmixing. *Int. J. Appl. Earth Obs. Geoinf.* 18:491-503. DOI `10.1016/j.jag.2011.10.004` — 遥感同型先例（端元近共线→求逆不稳→abundance 方差放大，用 VIF；不是 KS）
- ✅ Chang & Heinz 2001. Fully constrained least squares LSMA. *IEEE TGRS* 39(3):529-545. DOI `10.1109/36.911111` — FCLS：单纯形约束 abundance 估计遥感算法对应；有工程但缺解析条件数判据

---

## 3. 桥接引理可行性裁断（诚实评估）

**结论：不是套现成不等式，原创工作集中在"KS→几何"那一步的方向问题。但可证、中等偏低难度，且"证不出通用版"本身是卖点。**

- **桥的两端 = 现成砖**：几何→代数（夹角→条件数，Horn-Johnson 现成）；高斯间距→Gram（重叠积分闭式，Matrix Cookbook）；可识别性根基（Yakowitz-Spragins）。
- **中间一步 = 真 gap（需原创）**：KS(ρ)→L2/角度的**反方向界一般不存在**（Gibbs-Su/Gaunt-Li/Sason-Verdú 全是 KS≤f(其他)）。M1 失效恰走"ρ 小 ⇒ Gram 近奇异"这条没有通用 arrow 的反方向。
- **解法 = 走高斯特解**：高斯端元下 CDF sup-norm 距离 ρ 与峰间距 Δμ 有单调闭式关系，密度有界 M=1/√(2πσ²)，可在高斯受限族里把 ρ→Δμ→overlap→λ_min 串成单侧解析界。**这一串的拼装 + ρ↔Δμ 显式常数，是 §3 唯一真正要自证的原创引理。**
- **难度**：中等偏低、确定可做。"通用反向界不存在、必须用高斯结构"反而抬高理论站位。**红线：论文里不能宣称"直接引用现成不等式"，必须老实说这步原创。**

> **2026-06-11 更新（引理1 v0.2 对抗自检后）**：桥接引理已草拟并过一轮 6-agent 对抗自检（详 [`lemma1_draft.md`](lemma1_draft.md)）。代数闭式 (i)(ii)(iii) 机器精度验证全对；完整 R 的 κ~ρ^{−2(K−1)} 指数确切。**一处 serious-flaw 已修**：算法实际面对的病态在 sum-to-one **单纯形切空间**上，指数比完整 R **低 1 个幂次** → **κ_simplex ≍ ρ^{−2(K−2)}**（K=2 时切空间 κ≡1，本就良态）。故论文里"算法病态"的标度用 **ρ^{−2(K−2)}**，完整 R 的 ρ^{−2(K−1)} 只作密度几何陈述。整条因果链闭合仍依赖 Lemma 2（L²→Fisher 谱等价，本轮未证）。
- **另一 gap**：有限 N 颗锆石多项式噪声 × 病态 Gram 的联合**非渐近**误差界，无现成定理，可能要自拼矩阵 Bernstein + Higham 扰动界。

---

## 4. 差异化裁断（护城河）

四要素逐一核对：

| 要素 | 是否被先例占 |
|---|---|
| DZ 年龄谱为载体 | 已被大量占（Sundell-Saylor/Sharman-Johnstone/Tipton/Graham），**只是背景，不是差异点** |
| known-H（字典固定）设定 | ⚠️ **危险**：Sharman-Johnstone/Sundell-Saylor/Tipton 都是 known-parent，known-H 在地学早是标准。**新意绝不能停在"假设 H 已知"** |
| KS 度量驱动失效 → Gram 条件数 | ✅ **真空白**：无人把 KS 距离 ρ 解析连到 Gram 最小奇异值 |
| K×ρ 经验失败相图 → 解析判据 | ✅ 空白（待 Dietze 2022 全文确认无偷偷给阈值）|

**定位（一句话）**：M1 新意 = 在已知端元 + KL-NMF 设定下，经一座 **KS 距离 ρ → 端元 Gram 条件数的解析桥接引理**，给出可识别性的**解析判据**（非经验相图/蒙特卡洛/后验采样）。§3 引言须：(a) 承认 known-H 不新、点名 Sharman-Johnstone/Sundell-Saylor/Tipton 为先例；(b) 把 Graham 2025 + Saylor 2019 划进 blind 谱系区分；(c) 把 Dietze 2022 定位成"同行已识别但仅经验、无解析判据"的缺口证据。

---

## 5. §3 命题骨架（可直接下笔）

**设定**：年龄轴网格上给定 K 个已知高斯端元密度 h₁…h_K（峰位 μ_k、带宽 σ_k），混合 g=Σw_k h_k，w 在概率单纯形上；观测为每样本 N 颗锆石多项式抽样；反演用固定字典 H 的 KL 散度乘法更新；端元相似度用 CDF sup-norm 距离 ρ_{ij}=‖CDF_i−CDF_j‖_∞（KS）。

**引理 1（桥接引理 · 原创内核）**：高斯端元族下，峰间距 Δμ_{ij} 与 ρ_{ij} 一一对应；存在显式 c(σ,M) 使 Gram HHᵀ 非对角元被 ρ 控制，进而 λ_min(HHᵀ) ≥ Φ(ρ; σ, K)（经 Geršgorin + 重叠积分闭式），ρ_min→0 时 Φ→0。*（自证；文献只给两端积木。）*

**引理 2（局部曲率 = 加权 LS/Fisher）**：KL-NMF 不动点处，w 在单纯形切空间（Σδw=0）的 Fisher 信息 = Hᵀdiag(1/μ)H，μ=Hw，条件数与 HHᵀ 同阶。*（GLM 搬运 + 切空间投影自证。）*

**主定理**：若 ρ_min ≤ ρ*(K,σ)，则 (i) w 近不可识别（引理1 + Yakowitz-Spragins）；(ii) 有限样本误差 E‖ŵ−w‖² ≲ κ(HHᵀ)/N。*（有限样本部分 = 多项式噪声 × 病态 Gram，需自证。）*

**推论（相图解析化）**：固定间距等带宽下 κ(HHᵀ) 随 K 超线性增长 → K×ρ 失败阈值曲面，解析复现经验相图（K=5,ρ=0.01→100%）。

> **起草顺序**：先写**引理 1**（全篇唯一真原创证明、审稿人最会盯）→ 引理 2 + 主定理代数部分（组装现成结果）→ 推论接数值相图。

---

## 6. 投稿前潘必须一手核的三处（已标 could-not-verify，不替打包成确定）

1. **Hall & Titterington 1985 / Hall 1981** — 误差界渐近方差锚点，调 JRSS-B 原文核卷期页 + 方差表达式。
2. **Dietze & Dietze 2022（Sedimentology, sed.12929）全文** — 差异化缺口证据，确认未给 KS/条件数阈值。
3. **Graham et al. 2025（Math. Geosci. 57:601-628）** — 最强竞品，核作者全名 + 页码。

---

*记录人：LN。本文件 = M1 理论线的单一真相源；后续起草进展续记于本文件 §7（新建）。*
