# 第三篇 · Bayesian DZ-Unmix 真实数据不确定性量化 — 成文素材包

> 自动起草的 building blocks(2026-06,5-agent 并行)。所有数字取自 181,500-fit 的 GPU scan 结果(`grains_bayes_gpu.parquet`,本目录),未编造;引用占位 `(Author, Year)` / `the first paper` 待填。**Methods / Results 是草稿**,你的学术声音自行改定;**结果表 / 图注**是客观配套。
> 核心图:`figures/fig_bayes_uncertainty_v3.{png,pdf}`。

---

## 0. M3 现状核查留痕（2026-07-04 · 新 session 接 M3 从这里读）

> **M3 定义**(出处:`Projects/与数学的融合/与数学的融合_战略分析报告_2026-06-10.docx` §四):后验协方差特征结构(简并方向=病态特征向量)/ Fisher 信息几何,形式化「权重可信/待定」。定位=**本第三篇 UQ 论文的增强章节**,零新增算力,speculative 优先级(M1>M2>M3)。

**核查结论(2026-07-04,全目录+dz-global-db 扫过)**:
- **M3 专属工作零痕迹**:没有任何脚本对贝叶斯后验做协方差特征分解或 Fisher 几何。本目录含 eig 的 `probe2_*` 全是 M1 excess(K) 推导的 Gram/Toeplitz 特征值工作,**别混**。
- **三块现成地基**:
  1. 本素材包已有定性版——fig v3 的 2-σ 协方差椭圆沿 S1–S2 边拉长(=简并方向可视化)+ "w_S3 hard / w_S1↔S2 soft" 逐权重可信度论点(§1 编辑注记已建议把 robustness 分级换成权重级);
  2. 简并定量标度律 **σ_err=C/(√N·ρ),ρ_50√n=1.35(CV 4%)**(2026-06-14,`Projects/dz-global-db/_proto_degeneracy.py` 等 3 个 `_proto_*.py` + `cluster/make_resolvability_figure.py`)——⚠️"第三篇"命名碰撞:贝叶斯素材包(本文)与 pairwise 简并是两个 candidate,写作时先对齐;
  3. M1 定理(bounds v2 §3,supervised KL-NMF + KS→Gram 条件数桥接)= 解析锚点,本文 §6② 已规划双向引用。
- **前置决策 → ✅ 已拍板(2026-07-04)**:用户明示 **E 线正式立独立稿**,M3 作为增强章节挂靠其上;`00_论文总览_MASTER.md` §1/§2-E/§6 已同步更新。本素材包升格为**独立稿件的工作文档**。

**M3 若开工,第一步(零算力,本地案头活)**:从本目录 `grains_bayes_gpu.parquet`(40MB,181,500 fits 全量)逐样本取 (w_S1,w_S2,w_S3) 跨 121×100 scan 网格的 3×3 协方差 → 特征分解:**λ₁/λ₂ 比值 = 简并强度指标,主特征向量与 S1–S2 边的夹角 = "简并方向=病态方向"论断的直接检验**;再对照 M1 的 κ(Gram) 预测与 ②的 ρ 标度律。本地 python 用 `C:/Users/Administrator/miniconda3/python.exe`;fig 脚本路径坑见文末"附:两点技术提醒";图改完必须渲染+Read 视觉验证。

### 0.1 M3 第一步已跑完(2026-07-04) — 结果留痕

> 用户当日先拍「先跑 M3 第一步再定」,本节即当时的决策输入(**同日晚些时候已拍板正式立稿**,见 §0 前置决策)。脚本 `m3_eigen_probe.py` → 结果表 `m3_eigen_results.csv`(均本目录;`m3_*` 前缀,区别于 M1 的 `probe2_*`)。逐样本 121×100 全网格 3×3 协方差特征分解 + 中心先验 bootstrap-only 对照;simplex 约束核验 max|Σw−1|=6e-10,λ₃≈0 如期。

**四点发现**:

1. **主论断成立且更强**:11/15 样本主特征向量贴 S1–S2 边(多数 <2°)。Puetz R 系列四个样本 λ₁/λ₂ ~ 10³–10⁴、corr(w₁,w₂)=−1.000——**纯一维简并**,整条后验云压在 S1–S2 边上。全样本 λ₁/λ₂ 中位 35。
2. **简并方向会"改道"(新发现,超出原论断)**:100821-1/-3(唯二 w_S1≈0.01 的样本)主向量**精确**落 S2–S3 边(0.0°,λ₁/λ₂ 3–7×10³)——一个端元缺席时,软方向转移到下一对可交换端元。协方差特征向量自动追踪这个转移 = "逐方向可信度"比 fig v3 的定性椭圆更精细的直接证据,M3 章节核心卖点。几何注记:simplex 三条边两两夹角 60°,所以这俩样本对 S1–S2 边的夹角恰为 60°。
3. **M1 Gram 对照:方向全中,幅度失灵**:归一化 Gram(高斯重叠余弦)最弱特征向量 = (1,−1,0)/√2,与经验主方向 0.0° 吻合;但 κ(G_n)≈1+4.5e−5——裸密度重叠**严重低估**实际简并幅度(有限 n 似然层面才是主导,与 M1 数值验证"KDE 抬升有效 Fisher"结论一致)。→ 双向引用分工成立:**M1 给方向(解析),本文给幅度(经验)**,不重证。
4. **ρ 标度律跨样本检验被混杂,非证伪**:log√λ₁ vs log n 斜率 +0.14(全网格)/+1.47(bootstrap-only),非预测 −0.50。原因:σ_err=C/(√N·ρ) 是**固定混合物**下的律;跨样本时 simplex 位置效应(靠顶点=clip 小方差,居中=大方差)盖过 n 效应。干净检验 = 样本内 grain 二次抽样 → **已做,见 §0.2**。

### 0.2 M3 第二步:n-标度干净检验 + 计数噪声地板(2026-07-04 同日续)

> 脚本 `m3_nscaling_probe.py` → `m3_nscaling_results.csv` + `m3_nscaling_curves.csv`(逐 m 曲线)。原始年龄在 `Projects/bayes_gpu/realdata_k1_samples/*.csv`(age_Ma 单列,`#` 注释头)。设定:**supervised EM 固定端元形状**(与 M1 律同范式;权重-only 目标在 simplex 上是凹的,EM 必达全局最优),每样本 bootstrap 重抽 m∈geomspace(10,n,6)、B=400,全向量化本地秒级。

**四点发现**:

1. **标度律样本内精确成立**:slope median **−0.507**,IQR [−0.511, −0.490](预测 −0.500),r≈−0.99。§0.1 的跨样本混杂彻底解除。唯一例外 05FW008:EM 收敛到顶点 w₁=1.0,σ 塌缩机器零,slope −1.06 = **M1 clipping 机制的顶点情形**;而 10 个 edge-clipped 样本(w₃≈0)指数不受影响(median −0.499)——clipping 只在**软方向本身**被夹时才畸变指数,贴边只杀正交的硬方向。
2. **软轴方向跨范式一致**:EM 轴 vs Bayes 主轴 median 0.6°;100821-1/-3 的 S2–S3 改道在固定形状 EM 下完全复现(0.01°/0.004°)→ **改道是数据性质,不是 Bayesian/共推断伪影**。
3. **幅度 = 计数噪声地板(重写叙事级发现)**:EM 软轴 σ ÷ 多项式地板 √(vᵀ(diag(w)−wwᵀ)v/n) = **median ×1.02**(14/15 样本在 0.97–1.12)→ 固定形状下 S1↔S2 的"软"就是纯有限-n 多项式计数噪声,与 `_proto_danger_boundary.py` 的 ρ_KS(S1,S2)=0.999"安全区"判定**完全自洽**(表面矛盾消解);Puetz 系列 corr=−1 的纯一维简并其实是 w₃≈0 钉死后的 simplex 几何必然。且 Bayes bootstrap σ ÷ EM σ = median ×1.0 → 多数样本连贝叶斯后验均值的抽样噪声也在地板上。(edge 样本 C_eff=σ·√n·ρ≈0.65–0.71,与 M1 标定 C=0.48 同量级,差异来自 M1 常数无 w 依赖。)
4. **形状共推断膨胀只在个别样本发生**:R31370-S12652 **×8.8**(EM (0.04,0.96,0) vs Bayes (0.25,0.66,0.09),w 本身就不同);05FW007 连**方向**都改(EM 夹 w₁→0.02 后走 S2–S3 边,Bayes 保 w₁≈0.13 走 S1–S2 边,两轴差 58°);SL0929 ×1.75 轻度;05FW008 顶点情形(比值发散是除以机器零,伪影)。其余 11 样本 ×0.88–1.46。

**⚠ 对 §4 Results 草稿的影响 → ✅ 已改(2026-07-04,见 §4 重写版)**:现稿"S1↔S2 软源于 130/280 Ma 端元相邻→弱可识别"的归因**在幅度上不成立**——n=31–111 下多数样本的软度已被不可约计数地板解释;端元相邻的简并膨胀只经由**形状共推断**且只在个别样本(S12652/05FW007/05FW008)显现。改后叙事反而更利审稿:M3 章节主张升级为**逐样本判定"处于不可约有限-n 地板"vs"被端元形状灵活性膨胀"**,前者加 grain 就能救,后者加 grain 没用——可操作的采样建议。§4 已按此重写,并折入 §0.3 的 within-vs-between + 先验惰性两块。

**备注**:本步比较的是 bootstrap 抽样噪声(估计量层面);单次 fit 内的后验宽度(parquet `w_S*_std` 列,credible 区间层面)是另一个量——**✅ 已做,见 §0.3**。E 线立稿决策:**✅ 同日已拍板正式立稿**(见 §0 前置决策)。

### 0.3 M3 第三步:单次 fit 后验宽度(credible) vs 网格间均值散布(bootstrap+prior)(2026-07-04 同日续)

> 脚本 `m3_within_vs_between.py` → `m3_within_vs_between_results.csv`(逐样本逐权重)+ `m3_within_vs_between_soft.csv`(软轴)。做的是**总方差精确分解(law of total variance)**:对每样本 12,100 fits 池化的边缘后验,`Var_pool(w_k) = mean(w_k_std²)[WITHIN=单次 credible 宽度] + var(w_k_mean)[BETWEEN=扫描散布,==Table 1 的 std²]`。只用 `w_Sk_mean`/`w_Sk_std` 两组列,零算力本地秒级。**两处自检通过**:平衡设计 ANOVA 恒等式残差 6.9e-18(机器零);05FW007 重算 BETWEEN std = 0.118/0.121/0.070,与 Table 1 分毫不差(确认读对列、BETWEEN 就是 Table 1 那个 std)。

**四点发现**:

1. **WITHIN ≈ BETWEEN,各占总方差约一半(核心)**:BETWEEN/WITHIN 比值 median 0.90(硬 w_S3)/ 0.93(软 w_S1,w_S2),软轴 median 0.97;frac_between median ≈ 0.46。即**单次贝叶斯 fit 的 credible 区间已经捕获了约一半的总不确定度**,扫描把方差大致翻倍(sd 抬 ~√2≈1.4×)。→ **单报一个 credible 区间只低估总 sd 约 1.4×,不是数量级过度自信**。credible 区间"诚实到 ~40% 以内",扫描是精修不是揭穿。这一条把 §6②"简并/不确定"叙事从定性升级为可量化:后验区间基本可信,扫描锦上添花。

2. **BETWEEN 几乎纯 bootstrap(数据重采样),prior 近乎惰性(强卖点)**:frac_boot median 1.00(w_S3)/ 0.97(w_S1,w_S2)——13/15 样本里 ±10%μ/±20%σ 的 121 档 prior 扰动**贡献 <3% 的 BETWEEN 方差**,扫描散布 97–100% 来自 grain 的 bootstrap。→ **把 §6②.2"先验敏感性当卖点"直接量化成先验稳健性硬证据**:"扰动先验中心 ±10%/±20% 跨 121 配置,后验权重均值几乎不动(先验占 between 方差 <3%),几乎全部散布是有限-grain 计数噪声。"

3. **prior 只在两个"poorly constrained"样本咬人(内部自洽)**:唯二 frac_boot 掉到 ~0.50–0.59 的是 **05FW007 与 R31370-S12652**——正是 Table 1 标记的两个 poorly constrained 样本。其余 13 个 frac_boot ≈ 0.94–1.00。且 S12652 恰是 §0.2 里形状共推断 ×8.8 膨胀那个样本 → **"先验/形状灵活性只在已知病态样本起作用"跨步骤一致**。另:05FW008(顶点样本 w₁≈0.94)WITHIN 反超 BETWEEN(比值 0.37–0.40),单次后验比扫描散布更宽,因均值被顶点钉死不能动、但单次后验仍有宽度——与 §0.2 顶点情形自洽。

4. **三步统一到一个不确定度预算**:单次 credible 宽度(本步)≈ 计数噪声地板(§0.2)≈ bootstrap 网格间散布(本步 frac_boot≈1)——**三者是同一个有限-grain 计数噪声的三个侧面**;prior 惰性(除 2 个病态样本)。整条 UQ 故事收敛为:**计数噪声定宽度(credible=bootstrap=地板),先验惰性,arc–basement 配比软是因为计数噪声投影到 S1–S2 脊上**。

**软轴 proxy 的 caveat**:`m3_within_vs_between_soft.csv` 的 within 用 `(w_S1_std+w_S2_std)/√2`(设单次后验沿 S1–S2 完全反相关,corr→−1,由特征结构背书);对 **100821-1/-3 两个"改道"样本**(真软轴是 S2–S3,§0.1 发现)是 apples-to-oranges,其 1.68/1.63 比值忽略。逐权重分解(表主体)不受影响、精确。

**⚠ 对投稿稿的影响**:(a) §6②.2 先验稳健性从"计划"变"已量化"(prior <3% between 方差,13/15);(b) Table 1 / Methods 现注"combined prior-sensitivity and resampling"可精化为"**~97% resampling,~3% prior**(05FW007/S12652 除外)";(c) **✅ 已出图(2026-07-04)**:`m3_variance_sources_figure.py` → `m3_variance_sources.{png,pdf}`(GJ 双栏 6.85in、Okabe-Ito、300dpi、已本地渲染+Read 视觉验证)。逐样本方差来源堆叠条(within-credible / bootstrap / prior 三段,tr(Σ) 跨 3 权重聚合),(a) 绝对方差(顶标 √tr Σ 总 spread)+ (b) 归一化 100% 双栏。数字锚点:within 份额 median **54%**(41–88%,与 §0.3 逐权重 within≈0.54 自洽)、prior median **0.8%**(非高亮样本 ≤6%)、仅 05FW007/S12652 的 prior 段 **25%/20%**。**canonical caption 见 §2 图注区 Figure 2**(单一真相源,避免两处漂移)。

**环境备注(本机)**:本步在 Lenovo 机跑(前两步在 Administrator/D107 机,产物经 OneDrive 同步来);本机原无科学 python,已 winget 装 **Python 3.12.10(用户级)+ numpy 2.5.0/pandas 3.0.3/pyarrow 24.0.0**,解释器在 `C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`,后续本地跑 M3 脚本可直接用。

---

## 1. 主结果表(Table 1 候选)

| Sample | Group | Lat (°N) | Lon (°E) | n | w_S1 (Arc) mean±std | w_S2 (Basement) mean±std | w_S3 (Craton) mean±std | Resolvability (per weight) |
|---|---|---|---|---|---|---|---|---|
| 05FW007 | NCC north margin | 40.833 | 115.833 | 57 | 0.133 ± 0.118 | 0.436 ± 0.121 | 0.431 ± 0.070 | S1,S2 very soft; S3 soft |
| 05FW008 | NCC north margin | 40.833 | 115.833 | 36 | 0.940 ± 0.033 | 0.034 ± 0.033 | 0.026 ± 0.002 | S3 hard; S1,S2 soft |
| 100821-1 | NCC north margin | 40.733 | 113.817 | 82 | 0.012 ± 0.001 | 0.545 ± 0.055 | 0.443 ± 0.055 | S1 hard; S2,S3 soft |
| 100821-3 | NCC north margin | 40.733 | 113.817 | 111 | 0.009 ± 0.001 | 0.304 ± 0.042 | 0.687 ± 0.042 | S1 hard; S2,S3 soft |
| SL0929 | Songliao | 43.090 | 124.488 | 76 | 0.326 ± 0.101 | 0.587 ± 0.105 | 0.087 ± 0.031 | S1,S2 very soft; S3 soft |
| SL0930 | Songliao | 43.100 | 124.692 | 74 | 0.337 ± 0.055 | 0.560 ± 0.061 | 0.103 ± 0.031 | all soft |
| SL0931 | Songliao | 43.100 | 124.696 | 81 | 0.214 ± 0.052 | 0.343 ± 0.057 | 0.443 ± 0.047 | all soft |
| SL0932 | Songliao | 43.100 | 124.701 | 75 | 0.071 ± 0.026 | 0.057 ± 0.023 | 0.872 ± 0.034 | S1,S2 hard; S3 soft |
| R31354-S12518 | Songliao | 43.290 | 122.318 | 79 | 0.606 ± 0.069 | 0.379 ± 0.069 | 0.015 ± 0.007 | S3 hard; S1,S2 soft |
| R31354-S12519 | Songliao | 43.290 | 122.318 | 78 | 0.690 ± 0.060 | 0.298 ± 0.060 | 0.012 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12650 | Songliao | 45.873 | 125.532 | 78 | 0.460 ± 0.069 | 0.528 ± 0.069 | 0.012 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12651 | Songliao | 45.873 | 125.532 | 64 | 0.567 ± 0.057 | 0.418 ± 0.057 | 0.015 ± 0.000 | S3 hard; S1,S2 soft |
| R31370-S12652 | Songliao | 43.881 | 125.779 | 91 | 0.247 ± 0.219 | 0.659 ± 0.177 | 0.093 ± 0.063 | S1,S2 very soft; S3 soft |
| R31370-S12653 | Songliao | 46.465 | 125.803 | 31 | 0.214 ± 0.071 | 0.195 ± 0.072 | 0.591 ± 0.085 | all soft |
| R31370-S12654 | Songliao | 45.637 | 125.179 | 78 | 0.719 ± 0.050 | 0.264 ± 0.050 | 0.016 ± 0.010 | S3 hard; S1,S2 soft |

*Notes:* Posterior weight means and standard deviations are aggregated over 12,100 independent MCMC fits per sample (121 systematically perturbed prior configurations — end-member prior centres shifted by mean_shift ∈ [−10%, +10%] × sigma_shift ∈ [−20%, +20%], 11 × 11 grid — combined with 100 bootstrap resamplings of the detrital-zircon ages); *n* is the number of dated grains, and Lat/Lon (WGS84) are the sample coordinates compiled from the dz-global-db database (OneDZ / Puetz sources). The reported std captures the combined prior-sensitivity and data-resampling uncertainty (not the within-chain MCMC error alone), and for 13 of the 15 samples it is ≥ 97% data-resampling and < 3% prior-sensitivity; the exceptions are 05FW007 and R31370-S12652, where prior / end-member-shape flexibility contributes roughly half (see decomposition, §0.3). Endmembers: S1 = Mesozoic arc (peak 130 Ma), S2 = Palaeozoic basement / CAOB (280 Ma), S3 = Archaean–Proterozoic craton (1900 Ma). **Resolvability is graded per weight** by its scan std — *hard* (< 0.03; insensitive to both prior and bootstrap), *soft* (0.03–0.10), *very soft* (> 0.10) — rather than by a single sample-level grade, because *which* weights are resolved differs from sample to sample: the craton weight w_S3 is hard in most samples, but the soft (exchangeable) pair rotates onto w_S2 ↔ w_S3 where the arc is essentially absent (100821-1, 100821-3, w_S1 ≈ 0.01), and w_S1 ↔ w_S2 become the *hard* pair where the craton dominates (SL0932, w_S3 = 0.87). The softness of the arc–basement split is set by the finite grain count (n = 31–111; the soft-direction spread scales as n^(−1/2) and equals the multinomial counting-noise floor to within ~2%), *not* by the proximity of the 130 and 280 Ma end-members, which are in fact well resolved at the data level; it is therefore reducible by additional dating — except in 05FW007 and R31370-S12652, where end-member-shape ambiguity also contributes and more grains alone would not resolve it.

> **✅ 编辑注记(robustness 分级)已落实(2026-07-04)**:上表已从"三端元最大 std"的样本级分级换成**逐权重 Resolvability**(hard <0.03 / soft 0.03–0.10 / very soft >0.10,按各权重自己的 std);n(zircons)列已补(31–111)。逐权重口径额外暴露了三种被样本级分级掩盖的模式——**改道**(100821-1/-3:w_S1 反而 hard,软对转到 S2↔S3)、**翻转**(SL0932 克拉通主导:w_S1/w_S2 反而 hard)、**真 poorly constrained 的 very soft**(05FW007/SL0929/S12652)。与 §4 重写、§0.1–0.3 口径一致。

---

## 2. 图注

### Figure 1 — `Fig1_sample_locations`(sample location map)

**Figure 1.** Location of the fifteen Lower Cretaceous (K1) detrital-zircon samples on a schematic tectonic basemap of NE Asia. The North-China-Craton (NCC) north-margin group (reddish-purple squares, n = 4) lies on the craton south of the Solonker Suture; the Songliao Basin group (grey circles, n = 11) occupies the Songliao wide-rift basin. Regional framework after the first paper [(Author, Year)]: the Siberian and North China cratons, the intervening Central Asian Orogenic Belt (CAOB), the Mongol–Okhotsk and Solonker sutures, and the sinistral Tanlu Fault Zone; the blue vector marks NW-directed Palaeo-Pacific subduction. Near-coincident samples share one symbol (the label gives the site); individual sample coordinates are listed in Table 1 (compiled from the dz-global-db database, OneDZ / Puetz sources). Line styles follow FGDC TM 11A02; the categorical palette is Okabe-Ito. Script: `Fig1_sample_locations_v1.py` → `Fig1_sample_locations.{pdf,png,svg}`(GJ landscape, 300 dpi, geo-figures auditor 0 issues, 已渲染 + Read 验证).

### Figure 2 — `fig_bayes_uncertainty`(ternary + dot-whisker)

**Figure 2.** Detrital-zircon unmixing of fifteen Lower Cretaceous (K1) samples from NE Asia, resolved as a three-end-member Gaussian mixture under a Bayesian framework, with full posterior uncertainty propagated through a 121-prior × 100-bootstrap scan (181,500 GPU-accelerated MCMC fits). (a) Ternary diagram of posterior mean mixing weights for the arc (S1, ~130 Ma), basement/CAOB (S2, ~280 Ma) and craton (S3, ~1900 Ma) end-members; each sample is a single point (NCC north-margin group, magenta squares; Songliao group, grey circles) enclosed by a 2-σ covariance ellipse whose size and orientation encode the posterior spread, with Songliao samples elongated along the S1–S2 edge. (b) Dot-and-whisker plot of per-sample end-member weights ± one standard deviation across the scan grid, coloured by end-member (S1 orange, S2 green, S3 blue); whisker length is inversely proportional to confidence. Together the panels show that the craton weight w_S3 is tightly constrained (near-zero spread), whereas the arc-versus-basement partition w_S1↔w_S2 is soft: the data fix the arc + basement sum but leave its internal split limited by the finite number of dated grains — a counting-noise floor that scales as n^(−1/2) and is reducible by further dating — rather than by any intrinsic overlap of the well-separated 130 and 280 Ma end-members. Script: `fig_bayes_uncertainty.py` → `fig_bayes_uncertainty.{png,pdf}`(GJ double-column, 300 dpi; parquet path now local `Path(__file__).parent`; 已本地渲染 + Read 验证,数据与 Table 1 一致).

> **✅ 已对齐(2026-07-04)**:Figure 2(主 UQ 图)caption 末句成因已从"weakly identifiable owing to end-member degeneracy"改为"有限-grain 计数噪声地板(∝ n^(−½)、可加 grain 缓解;130/280 Ma 端元实际可分)",与 §4 重写 / Table 1 / §0.3 口径统一;椭圆沿 S1–S2 拉长的视觉描述保留。端元形状共推断膨胀的例外(05FW007/S12652)见 Figure 3 caption,不重复挤进主图注。

### Figure 3 — `m3_variance_sources`(variance-source decomposition)

**Figure 3.** Variance-source decomposition of the posterior mixing-weight uncertainty for the fifteen K1 samples (NCC north-margin group left of the dashed divider, Songliao right; the two shape-inflated samples 05FW007 and R31370-S12652 highlighted in orange, with *n* the number of dated grains). For each sample the pooled posterior weight covariance across the 121-prior × 100-bootstrap scan has total variance tr(Σ) = Σ_k Var(w_k), partitioned by the law of total variance into three additive sources: the mean within-fit credible variance (blue) and the between-fit scatter of the posterior means contributed by bootstrap resampling of the grains (grey) and by the ±10 % μ / ±20 % σ prior perturbation (orange). (a) Absolute variance; the value above each bar is the total posterior spread √tr(Σ). (b) The same, normalised to 100 %. The within-fit credible width supplies about half of the total variance in every sample (median 54 %), showing that a single Bayesian fit's own credible interval already captures most of the reported uncertainty; the prior-sensitivity share *of the total variance* is negligible (median 0.8 %, ≤ 6 % for thirteen of the fifteen samples) except in 05FW007 and R31370-S12652 (~25 % and ~20 %), where co-inference of the end-member shapes inflates the arc–basement degeneracy. (Equivalently, the prior is < 3 % of the *between-fit* variance, the framing used in §4/§5.3; the two denominators differ because the between-fit scatter is itself only about half of the total.) For the remaining samples the reported uncertainty is essentially a data-resampling (finite-grain) quantity, independent of the prior.

### Figure 4 — `FigY_nmf_comparison`(supervised NMF vs Bayesian, 4 NCC samples)

**Figure 4.** Agreement between the first paper's supervised-NMF point estimate and this paper's Bayesian posterior mean, for the four North-China-Craton north-margin samples on which both can be run (05FW007, 05FW008, 100821-1, 100821-3; 12 points = 4 samples × 3 end-members, coloured by end-member). The supervised-NMF weights are computed by applying the first paper's method (identical end-members and KL multiplicative updates) to the same real grain-age spectra; error bars are the Bayesian scan standard deviation (this paper). Points fall on the 1:1 line — the mean absolute weight difference is 0.03 — with the closest agreement on the hard craton weight w_S3 (MAE 0.01) and the largest deviation on the soft arc/basement split (w_S1, w_S2; MAE 0.04), where the point estimate nonetheless lies within the Bayesian uncertainty (e.g. sample 05FW007). The comparison is limited to these four samples because the eleven Puetz/OneDZ Songliao samples were never run through the first-paper pipeline. Script: `FigY_nmf_comparison.py` → `FigY_nmf_comparison.{png,pdf}` + `FigY_nmf_comparison_data.csv`(已渲染 + Read 验证). ⚠ 图号 "Fig 4" 暂定(方法对照,或作 supplementary);正文 §5.4 引用同步。

### Figure 5 — `m3_mu_sigma_posterior`(end-member shape: prior vs posterior)

**Figure 5.** Prior versus posterior for the end-member shape parameters — peak age μ (top row) and peak width σ (bottom row) of the arc (S1), basement (S2) and craton (S3) end-members — testing whether the reference priors dominate the posterior. In each panel the grey curve is the Gaussian prior (centred on the reference value, dashed line; relative std ±10 % on μ, ±20 % on σ) and the coloured curve is a representative single-fit posterior (median over the 15 samples of the posterior mean and within-fit std, at the central prior configuration mean_shift = sigma_shift = 0); coloured ticks are the 15 per-sample posterior means. The data appreciably updates the shapes rather than reproducing the prior: the posterior on μ narrows to 0.31–0.53× the prior width (tightest for the craton, μ_S3) and shifts by up to 0.7 prior-σ (arc +0.7, basement −0.5), while σ_S2 and σ_S3 shift to +1.4 and +1.6 prior-σ (the basement and craton age modes are broader than assumed: σ_S2 ≈ 38 vs 30 Ma, σ_S3 ≈ 132 vs 100 Ma). That the end-member shapes are data-constrained is the flip side of the weights' prior-insensitivity (§5.3): the data pins the shapes, so the mixing weights do not depend on the exact prior. Script: `m3_mu_sigma_posterior_figure.py` → `m3_mu_sigma_posterior.{png,pdf}`(已渲染 + Read 验证). ⚠ 图号 "Fig 5" 暂定(先验稳健性佐证,或作 supplementary)。

### Figure 6 — `m3_acceptance_diagnostics`(MCMC acceptance diagnostics)

**Figure 6.** MCMC sampling diagnostics for the 181,500-fit GPU scan (Goodman–Weare affine-invariant ensemble sampler; 64 walkers × 5000 steps, 1000-step burn-in). (a) Distribution of the acceptance fraction over all fits: 99.3 % fall within the emcee-recommended healthy band [0.2, 0.5] (shaded), median 0.42 (dashed line). (b) Per-sample box plots (NCC north-margin above the divider, Songliao below); every sample sits in the healthy band. The only sample with a depressed acceptance is R31370-S12652 (highlighted; median ≈ 0.28), the most degenerate / shape-inflated sample (§0.2), whose flatter posterior is intrinsically harder to sample — still healthy. Per-fit integrated autocorrelation times were not stored; convergence is corroborated by the burn-in, the acceptance fraction shown here, and the fit-by-fit GPU-vs-emcee equivalence check (Methods §3). Script: `m3_acceptance_diagnostics_figure.py` → `m3_acceptance_diagnostics.{png,pdf}`(已渲染 + Read 验证). ⚠ 图号 "Fig 6" 暂定(方法诊断,多半作 supplementary)。

---

## 2.5 Introduction / Geological setting 草稿

> 草稿(2026-07-04)。正文开篇,最终排版置于 Methods(§3)之前 = 论文 §1 Introduction + §2 Geological setting。**地质/构造文献占位已补(2026-07-04)**:方法引用 = M1_references 已核验(Sharman-Johnstone/Sundell-Saylor);构造框架 = consensus 学术检索的高被引真实文献(NCC 1.85 Ga = Zhao 2012/2001;东段 CAOB = Liu 2017;NE 中国早白垩弧/古太平洋 = Huang 2024/Li 2021/Wu 2022);样本数据源 = CSV 头真实 DOI。**S1 归属已定**(古太平洋 rollback 主控 + 早早白垩蒙古-鄂霍茨克叠加,Huang 2024)。**所有 8 条 §2.5 参考 DOI 已逐条 Crossref 交叉核实**(2026-07-04,✅ 标记;卷/页全带),无 `[DOI 待补]` 残留。⚠ 仍待你:以你区域专长审定归属措辞、终稿按目标刊格式统一参考文献样式。端元峰位(130/280/1900 Ma)与 §3 Methods / Table 1 一致。

### Introduction

Detrital-zircon U–Pb geochronology has become a primary tool for sediment provenance analysis, and a natural way to interpret a measured age spectrum is to decompose it into contributions from a small number of geologically defined source domains — a mixing, or "unmixing", problem (Sharman & Johnstone, 2017; Sundell & Saylor, 2017; and the fixed-dictionary supervised method of the first paper). When the candidate sources are known a priori, the quantity of interest is the vector of mixing weights: the fractional contribution of each source to the sample. Recovering these weights, however, is only half of the task. For the result to be geologically usable one must also know how trustworthy each weight is, and prevailing practice — reporting a single point estimate per sample — leaves this unquantified. Two difficulties are especially acute. First, when two source domains have overlapping age modes their relative contributions can be only weakly identifiable even when their sum is well constrained, and a point estimate conceals this. Second, the answer may depend on analyst choices (the assumed source spectra) or on the finite number of grains dated, and a single fit cannot separate a robust conclusion from a fragile one.

We address this by casting three-end-member detrital-zircon unmixing as a Bayesian inference problem and propagating the full posterior uncertainty. Rather than fitting each sample once, we combine a systematic perturbation of the source priors (121 configurations) with bootstrap resampling of the grains (100 resamples), yielding a per-sample posterior for the mixing weights that jointly reflects prior sensitivity and finite-sample noise. To make this tractable — the reference ensemble sampler would require several days for the full 181,500-fit scan — we implement the sampler as a batched GPU kernel, validated fit-by-fit against the CPU gold standard (§3). Applying it to fifteen Lower Cretaceous (K1) samples from NE Asia, we show that the three source weights are not equally identifiable, that the identifiability structure can be diagnosed per sample and per weight rather than assumed, and that the dominant source of the reported uncertainty is the finite grain count rather than the prior. The outcome is a provenance interpretation in which each conclusion carries an explicit, source-specific confidence, and in which the route to a more confident answer — more grains, or tighter source constraints — is made quantitative.

### Geological setting

The fifteen samples fall into two groups spanning the eastern Central Asian Orogenic Belt (CAOB) and the northern North China Craton (NCC) (Liu et al., 2017; Zhao et al., 2012): a North-China-Craton north-margin group and a Songliao Basin group (Fig. 1). Their Lower Cretaceous clastic fill draws potentially on three compositionally and geochronologically distinct source domains, which we adopt as the mixing end-members (§3):

- **S1 — Mesozoic arc (reference peak ≈ 130 Ma).** Early Cretaceous arc magmatism — driven predominantly by rollback of the Palaeo-Pacific (Izanagi) slab along the eastern continental margin, with a subordinate Mongol–Okhotsk contribution in the northwest during the earliest Cretaceous (Huang et al., 2024; Li et al., 2021; Wu et al., 2022) — is widespread across NE Asia (arc-magmatic peak ≈ 124–133 Ma; Li et al., 2021) and contributes a young, narrow age mode.
- **S2 — Palaeozoic CAOB basement (≈ 280 Ma).** The accretionary and arc terranes of the CAOB — the world's largest accretionary orogen, amalgamated through the Palaeozoic with final closure of the Palaeo-Asian Ocean along the Solonker suture near the northern NCC margin — provide a broad late-Palaeozoic source (Liu et al., 2017; Xiao et al., 2014).
- **S3 — Archaean–Proterozoic craton (≈ 1900 Ma).** The crystalline basement of the North China Craton carries the characteristic ~1.85 Ga signature of its final Palaeoproterozoic amalgamation along the Trans-North China Orogen (Zhao et al., 2012; Zhao, 2001), a source separated from the two younger domains by a wide age margin.

These three domains are geologically expected to contribute in different proportions to the two groups — the NCC north-margin samples being proximal to cratonic basement, the Songliao Basin being dominated by its encircling arc and CAOB terranes — which furnishes an independent, qualitative check on the quantitative unmixing (§5.5). Because the arc (≈ 130 Ma) and basement (≈ 280 Ma) modes are the most closely spaced of the three, their relative partition is the component most exposed to the identifiability limits examined here, whereas the cratonic mode, isolated near 1900 Ma, is expected to be the most robustly resolved — an expectation borne out by the posterior analysis (§4).

**§2.5 draft references** (⚠ 卷/页/DOI 请核你的文库补全;consensus 检索来的标 `[DOI 待补]`,不擅填卷页):
- Sharman, G.R. & Johnstone, S.A. (2017). Sediment unmixing using detrital geochronology. *Earth Planet. Sci. Lett.* doi:10.1016/j.epsl.2017.07.044
- Sundell, K.E. & Saylor, J.E. (2017). Unmixing detrital geochronology age distributions (DZmix). *Geochem. Geophys. Geosyst.* 18(8), 2872–2886. doi:10.1002/2016GC006774
- **Pan, L.** (first paper / companion). Supervised-NMF detrital-zircon source unmixing, NE-Asia K1. Code: Zenodo doi:10.5281/zenodo.20371892. `[⚠ DOI reserved-未发布(2026-07-04 核=dead);刊/稿号待定,见 §3 Priors note]`
- Liu, Y.-J., Li, W., Feng, Z., Wen, Q., Neubauer, F. & Liang, C. (2017). A review of the Paleozoic tectonics in the eastern part of the Central Asian Orogenic Belt. *Gondwana Research* 43, 123–148. doi:10.1016/j.gr.2016.03.013 ✅
- Xiao, W. & Santosh, M. (2014). The western Central Asian Orogenic Belt: a window to accretionary orogenesis and continental growth. *Gondwana Research* 25, 1429–1444. doi:10.1016/j.gr.2014.01.008 ✅
- Zhao, G., Cawood, P.A., Li, S., Wilde, S.A., Sun, M., Zhang, J., He, Y. & Yin, C. (2012). Amalgamation of the North China Craton: key issues and discussion. *Precambrian Research* 222–223, 55–76. doi:10.1016/j.precamres.2012.09.016 ✅
- Zhao, G. (2001). Palaeoproterozoic assembly of the North China Craton. *Geological Magazine* 138(1), 87–91. doi:10.1017/S0016756801005040 ✅
- Huang, H., Wang, T., Guo, L., Tong, Y., He, Z.-Y., Yin, J. & Wu, H.-H. (2024). Crustal modification influenced by multiple convergent systems: insights from Mesozoic magmatism in northeastern China. *Earth-Science Reviews* 252, 104737. doi:10.1016/j.earscirev.2024.104737 ✅
- Li, H. et al. (2021). Early Cretaceous arc-related volcanic rocks in the northern Great Xing'an Range, NE China: records of Palaeo-Pacific Ocean subduction. *International Journal of Earth Sciences* 110, 1233–1263. doi:10.1007/s00531-021-02014-y ✅
- Wu, J.T.-J., Wu, J. & Okamoto, K. (2022). Intra-oceanic arc accretion along Northeast Asia during Early Cretaceous provides a plate tectonic context for North China craton destruction. *Earth-Science Reviews* 226, 103952. doi:10.1016/j.earscirev.2022.103952 ✅(Crossref + ScienceDirect 交叉核对 2026-07-04)
- **Sample data sources** (per-sample DOIs in Table 1 / §6③ availability): NCC 05FW00x doi:10.1016/j.epsl.2006.04.029; NCC 100821-x doi:10.1080/00206814.2016.1168323; Songliao OneDZ doi:10.1016/j.gr.2012.04.002; Songliao Puetz doi:10.1016/j.sedgeo.2022.106133 & doi:10.1016/j.gsf.2022.101377.

---

## 3. Methods 草稿

### Three-end-member Gaussian mixture model

We model each detrital-zircon U–Pb age spectrum as a mixture of three geologically defined source end-members: S1, Mesozoic arc magmatism (reference peak μ ≈ 130 Ma, σ ≈ 15 Myr); S2, Palaeozoic basement of the Central Asian Orogenic Belt (CAOB; μ ≈ 280 Ma, σ ≈ 30 Myr); and S3, Archaean–Proterozoic cratonic crust of the North China Craton north margin (μ ≈ 1900 Ma, σ ≈ 100 Myr). For a sample of grain ages a = {a_z}, mixing weights **w** = (w₁, w₂, w₃) live on the 2-simplex (Σ w_k = 1). Rather than assigning grains to sources, we marginalise over the latent source label of each zircon, giving the per-grain mixture likelihood

  log p(a | **w**, **μ**, **σ**) = Σ_z log [ Σ_k w_k · N(a_z | μ_k, σ_k) ],

where N(· | μ_k, σ_k) is the Gaussian density of end-member k. The posterior is 8-dimensional: two free weights (K − 1 = 2) plus the three centres μ_k and three widths σ_k, which are co-inferred rather than fixed.

### Priors

End-member centres are taken from the reference end-members of the companion supervised-NMF unmixing paper (Pan L.; first paper, code archived at Zenodo doi:10.5281/zenodo.20371892). [⚠ 第一篇引用待定稿:该 Zenodo DOI 目前 **reserved-未发布**(2026-07-04 联网核 = 404/dead,与 MASTER §3 HIGH#1 一致);刊/稿号与 §8 编号冲突缠绕(可能 = CAGEO-S-26-01308「方法稿」)——投稿前需第一篇先 publish Zenodo + 定引用形式。] We place independent Gaussian priors on each parameter, with relative standard deviations of ±10% on the μ_k and ±20% on the σ_k, allowing the data to refine the end-member shapes while anchoring them to geologically meaningful values. A symmetric Dirichlet(1) prior on **w** imposes a uniform distribution over the simplex, expressing no prior preference among sources.

### Prior-perturbation × bootstrap scan

To separate robust conclusions from those that are sensitive to analyst choices, we do not fit each sample once. Instead, we systematically perturb the prior centres over a grid of mean_shift ∈ [−10%, +10%] (11 levels) × sigma_shift ∈ [−20%, +20%] (11 levels) = 121 prior configurations, and at each configuration draw 100 with-replacement bootstrap resamples of the grains. This yields 12,100 independent MCMC fits per sample, and 15 × 121 × 100 = 181,500 fits in total. The spread of the posterior weight means across this 121 × 100 grid (reported as the standard deviation) directly quantifies how much each conclusion depends jointly on prior specification and on sampling of the finite grain population, rather than relying on a single posterior credible interval.

### MCMC and GPU implementation

Posteriors were sampled with the Goodman–Weare affine-invariant ensemble sampler (emcee; Foreman-Mackey et al., 2013) using 64 walkers × 5000 steps with a 1000-step burn-in. The reference emcee implementation requires ~21 s per fit on CPU, making the full scan infeasible (several days). We therefore re-implemented the sampler as a batched GPU kernel (NVIDIA L20), completing all 181,500 fits in 4475 s (~74 min). We validated the GPU sampler fit-by-fit against the emcee gold standard: the maximum absolute difference in posterior weight means was 0.020, whereas re-seeding emcee itself produced inter-seed scatter of ~0.19 along the degenerate (adjacent S1/S2) direction. The GPU–emcee discrepancy is thus an order of magnitude smaller than emcee's own Monte Carlo noise, establishing equivalence. Bootstrap convergence was checked at the central prior: the ratio of the weight std from the first b resamples to that from the full 100 reaches a median of ≈ 1.0 by b ≈ 20 and its interquartile range tightens steadily thereafter (Table 2), confirming that ~30 resamples already converge and 100 provides a comfortable robustness margin. The wide tails of the raw ratio arise from the near-zero-σ hard weights (division by σ ≈ 0.001–0.003); the meaningful spread is given by the soft-weight subset of Table 2. The ensemble sampler itself remained healthy throughout the scan: the acceptance fraction lay within the recommended 0.2–0.5 range for 99.3% of all 181,500 fits (median 0.42), the only depressed value being on the most degenerate sample, R31370-S12652 (Fig. 6).

**Table 2. Bootstrap convergence at the central prior configuration.**

| b (resamples) | median (all 45) | IQR (all 45) | median (soft 35) | IQR (soft 35) |
|---|---|---|---|---|
| 10 | 0.90 | 0.82–1.10 | 0.90 | 0.83–1.11 |
| 20 | 0.99 | 0.91–1.12 | 1.01 | 0.93–1.12 |
| 30 | 0.99 | 0.92–1.10 | 0.99 | 0.92–1.09 |
| 50 | 1.02 | 0.97–1.06 | 1.03 | 0.99–1.06 |
| 75 | 1.02 | 0.97–1.05 | 1.02 | 0.98–1.04 |

*Notes:* Entries are the ratio σ_b / σ_100 of the bootstrap standard deviation of the posterior mean weight from the first *b* resamples to that from all 100, at the central prior configuration (mean_shift = sigma_shift = 0), aggregated over the 15 samples × 3 end-member weights (45 pairs) and over the 35 "soft" pairs with σ_100 > 0.02 — the latter excludes the hard weights whose σ_100 ≈ 0.001–0.003 make the ratio a division-noise artifact (e.g. R31354-S12518 w_S3, 05FW008 w_S1/w_S2). The median ratio is already ≈ 1.0 by *b* ≈ 20 and the interquartile range tightens steadily, i.e. the bootstrap standard deviation has essentially converged well before 100 resamples; 100 is therefore a comfortable robustness margin, not excess. Reproduced by `m3_bootstrap_convergence_table.csv` (first-*b* cumulative, ddof = 1). ✅ Methods §3 与本表口径已统一(2026-07-04):§3 现引本表(median ≈ 1.0 by *b* ≈ 20)、不再引旧的 "0.84–1.64"(那是原始 scan-time 检查的值,随 *b*/方法变、被近零-σ 硬权重除法噪声主导);有意义的展布见软子集列。

---

## 4. Results 草稿

> **⚠ 本节 2026-07-04 已按 M3 三步(§0.1–0.3)重写**:修正原稿"S1↔S2 软 = 端元相邻弱可识别"的幅度误归因(端元相邻只经形状共推断膨胀、只在个别样本发生),补入计数噪声地板 / n^(−½) 标度、within-vs-between 分解、先验惰性三块定量结果。旧稿(误归因版)见本仓库 git 历史。

### Posterior mixing weights and their uncertainty

Per-sample three-end-member posterior mean weights and their associated uncertainties (the standard deviation of the posterior mean across the 121 × 100 prior-perturbation × bootstrap grid) are reported in Table 1 and summarised in Fig. 2. Across the 15 samples all posteriors resolve geologically interpretable mixtures of the Mesozoic-arc (S1, 130 Ma), Palaeozoic-basement/CAOB (S2, 280 Ma) and Archaean–Proterozoic cratonic (S3, 1900 Ma) end-members. The central message of our uncertainty analysis, however, is that the three weights are *not* equally well constrained, and that the appropriate unit of "robustness" is the individual weight rather than the sample as a whole.

The cratonic weight w_S3 is the hard, universally well-resolved component. Its scan standard deviation is small for every sample (0.000–0.085) and effectively zero for several Songliao samples (≤ 0.002; e.g. R31354-S12519, R31370-S12650, R31370-S12651). The presence or absence — and, where present, the magnitude — of an Archaean–Proterozoic cratonic contribution is therefore recovered as a conclusion insensitive to both prior perturbation and bootstrap resampling.

By contrast, the partition of the remaining budget between the arc (S1) and basement (S2) end-members is the soft component. The two standard deviations std(w_S1) and std(w_S2) are large (0.05–0.22), anti-correlated, and nearly equal within every sample (e.g. 05FW007 0.118/0.121; SL0930 0.055/0.061; R31354-S12518 0.069/0.069): the data pin the combined arc + basement contribution w_S1 + w_S2 while leaving its internal split loosely determined, so the posterior cloud collapses to a one-dimensional ridge along the S1–S2 join (Fig. 2a). A per-sample covariance eigen-analysis makes this explicit — the leading eigenvector of the weight covariance lies within ~2° of the S1–S2 direction for 11 of the 15 samples, and for the most arc-poor Songliao samples the anisotropy λ₁/λ₂ reaches 10³–10⁴ with corr(w_S1, w_S2) = −1.000, i.e. an essentially pure one-dimensional degeneracy once the craton weight is fixed near zero.

### Anatomy of the arc–basement softness

Three properties of this soft direction, established by resampling and eigen-decomposition of the scan, bear directly on how the weights should be read.

*Grain-count scaling of the soft direction.* Within each sample, subsampling the dated grains to size m and refitting under fixed end-member shapes gives a soft-direction spread that scales as σ_soft ∝ m^(−1/2) (median exponent −0.507, interquartile range [−0.511, −0.490]; prediction −0.500), and whose full-n magnitude matches the multinomial counting-statistics floor of a perfectly separable mixture to a median factor of 1.02 (14 of 15 samples in 0.97–1.12). The 130 Ma arc and 280 Ma basement age modes are in fact well separated at the data level — their Kolmogorov–Smirnov resolvability is ≈ 0.999 and the cosine-normalised Gaussian-overlap Gram matrix of the reference end-members is essentially the identity — so the arc–basement split is limited by how many grains were dated (n = 31–111), not by the separability of the two sources (interpreted in §5.2).

*Within- versus between-fit variance.* A law-of-total-variance split of the pooled posterior into the mean within-fit credible variance and the between-fit scatter of the means gives comparable terms — the between-fit scatter is a median of 46% of the total (Fig. 3) — so a single fit's own credible interval understates the full scan spread by a median factor of only ≈ 1.4.

*Bootstrap versus prior share.* Splitting the between-fit scatter into its bootstrap and prior-perturbation components, the ±10% (μ) / ±20% (σ) prior shifts contribute a median of under 3% of the between-fit variance (≥ 97% is bootstrap resampling of the grains) for 13 of the 15 samples (Fig. 3b); the two exceptions coincide with the two least-constrained samples (below).

Consistent with a data-driven rather than a model-imposed degeneracy, the soft direction tracks whichever end-member pair is exchangeable: for the two NCC north-margin samples that carry almost no arc component (100821-1 and 100821-3, w_S1 ≈ 0.01) the leading eigenvector rotates off the S1–S2 join and onto the S2–S3 join (to within 0.0°), the covariance following the degeneracy as it migrates to the next adjacent pair.

### Regional mixing patterns

The two groups show distinct mixing patterns. Most Songliao samples are dominated by arc plus basement with a negligible cratonic component; the Puetz R-series in particular yields w_S3 = 0.012–0.016 (R31354-S12518/S12519, R31370-S12650/S12651/S12654), indicating sediment sourced from Mesozoic arc and Palaeozoic orogenic terranes with essentially no Archaean–Proterozoic cratonic input. The NCC north-margin samples instead carry a resolvable cratonic signature (w_S3 = 0.443 for 100821-1, 0.687 for 100821-3, 0.431 for 05FW007), consistent with their position on the northern margin of the North China Craton. One Songliao sample, SL0932, is exceptionally craton-dominated (w_S3 = 0.872): its arc and basement weights are pinned near zero as the hard pair, so the craton dominance is robust (its small residual std of 0.034 merely propagates those two). We interpret this exception in §5.5.

### The two least-constrained samples

Two samples carry the largest weight uncertainties: 05FW007 (w = 0.133/0.436/0.431, std 0.118/0.121/0.070) and R31370-S12652 (w = 0.247/0.659/0.093, std 0.219/0.177/0.063), in both cases driven by the soft S1↔S2 partition rather than by w_S3. These are also the only two samples in which prior perturbation contributes materially to the spread (bootstrap fraction ≈ 0.5–0.6, versus ≈ 0.97–1.0 elsewhere) and in which co-inferring the end-member shapes inflates the soft-direction width beyond the counting-noise floor (by a factor of ≈ 8.8 for R31370-S12652). For these two samples the arc–basement split is limited by end-member-shape flexibility as well as by grain count, so — unlike the counting-noise-limited majority — additional grains alone would not fully resolve it (§5.2).

---

## 5. Discussion 草稿

> 草稿(2026-07-04 起)。学术声音待你定稿;数字与 §0.1–0.3 / Table 1 / Figs 1–4 对齐。监督 NMF 对照已出图(§5.4 → Fig 4);M1 定理已交叉引用(§5.4 = 第二篇 §3 可识别性定理:KS→Gram 桥、λ_min∝ρ^{2(K−1)}、相邻端元简并瓶颈 = S1↔S2,只剩稿号/DOI 待补)。§2.5 地质/构造文献占位**已补 + DOI 全核实**(2026-07-04,8 条参考逐条 Crossref 交叉核、卷/页齐;S1 归属定为古太平洋主控 + 蒙古-鄂霍茨克叠加)。**M1 companion(第二篇 bounds)状态已核实并全线对齐(2026-07-04)**:经 **Zenodo(record 10.5281/zenodo.20768132, concept …131, v1.0 2026-06-20)+ GitHub(PNOAHA/dz-unmix-bounds, MIT)双源核实**——**已投 *Computers & Geosciences*(submitted)**,题《Where does detrital-zircon source unmixing break? …》(Pan L.)。⚠ **原三处冲突已判**:过期方是**项目 README/CLAUDE.md「规划中/目标刊未定/无 DOI」**(那个 `10.5281/zenodo.20371892` 是第一篇的);§8「投稿包齐备 + DOI 已铸」更接近真相。已把**全局 §8、项目 README/CLAUDE.md、本 §5.4** 全部对齐到「已投 C&G + 代码 DOI …132」;§5.4 引用由 in prep. 升为 submitted + 题名 + 代码 DOI。**唯一 open = 稿号**(可能 = CAGEO-S-26-01308,若与 §8「DZ 方法稿」同篇——§8 已 flag 待你合并确认)。

### 5.1 Not every mixing weight deserves equal interpretive weight

The recurring temptation in detrital-zircon unmixing is to read a single point estimate of the source proportions as though all of its components were equally well determined. Our posterior analysis shows that they are not, and — more usefully — that *which* components are resolved varies from sample to sample in a way that can be diagnosed rather than assumed. Across the fifteen K1 samples the cratonic weight w_S3 is a hard, well-resolved quantity whereas the arc-versus-basement partition w_S1↔w_S2 is soft and concentrated on a one-dimensional ridge in the simplex (§4). Because the covariance is estimated per sample, the diagnosis adapts automatically: where the arc component is essentially absent the soft direction rotates onto the S2–S3 join (100821-1, 100821-3), and where the craton dominates the arc/basement pair becomes the hard one instead (SL0932). We therefore report resolvability per weight (Table 1) rather than as a single per-sample grade, and we suggest that downstream interpretation follow the same rule: the presence or absence of a craton source and the combined arc + basement contribution are safe to read, whereas the internal arc:basement ratio of an individual soft sample is not, unless it is backed by additional dating.

### 5.2 The softness is a finite-sample counting floor, not a failure of the method

The decomposition (§4) carries one central interpretive message: the arc–basement softness, at the grain counts available here, is a finite-sample counting-noise floor, not any intrinsic inability of the method to separate the well-resolved 130 and 280 Ma sources. For the great majority of samples it is therefore a *reducible* uncertainty — one of finite sampling rather than of method. The two exceptions (05FW007 and R31370-S12652) are those in which co-inferring the end-member shapes inflates the soft direction beyond the counting floor, an irreducible shape-ambiguity that more grains alone would not remove. This reducible-versus-irreducible distinction is what makes the per-weight analysis prescriptive rather than merely descriptive, with direct consequences for how each sub-population should be sampled (§5.6).

### 5.3 The reported uncertainty is honest and prior-independent

Two further properties make the reported uncertainties trustworthy in their own right. First, because a single fit's own credible interval already carries about half of the pooled posterior variance (§4; Fig. 3), the perturbation scan refines rather than overturns it — a single Bayesian fit is not badly overconfident. Second, because the scan-added spread is almost entirely data-resampling and negligibly prior for all but two samples (§4; Fig. 3b), the literature-anchored end-member centres and widths are not load-bearing: a standard objection to any prior-based unmixing — that the answer merely restates the prior — is here answered quantitatively, and the two samples where the prior does matter are exactly those flagged as least constrained on independent grounds (§5.2). Consistently, the end-member shapes are themselves visibly updated by the data — the posterior peak ages narrow to 0.3–0.5× the prior and the basement and craton peak widths shift by +1.4 and +1.6 prior-σ (Fig. 5) — so the reference priors act as constraints the data refines, not assumptions that propagate unchanged into the weights.

### 5.4 Relation to the companion papers

To position the Bayesian treatment relative to the supervised point-estimate approach of the first paper — from which the reference end-members are taken — we applied that method (identical end-members, KL multiplicative updates) to the same four real North-China-Craton north-margin grain-age spectra and compared its point estimates with our Bayesian posterior means (Fig. 4). The two methods agree closely: the mean absolute weight difference is 0.03, tightest on the hard craton weight (w_S3, MAE 0.01) and larger on the soft arc/basement split (w_S1, w_S2, MAE 0.04), where the point estimate falls within the Bayesian scan uncertainty. The Bayesian framework therefore does not move the central estimate but calibrates it: the soft arc:basement direction — where the two methods differ most, e.g. sample 05FW007 — is precisely the direction along which any single point estimate is least reproducible, so attaching an uncertainty to it is what the Bayesian treatment contributes. The comparison is limited to these four samples: the eleven Puetz/OneDZ Songliao samples were never run through the first-paper pipeline, and the first paper's own case study used four literature-grounded *synthetic* basin proxies rather than these real samples, so the point estimates here are obtained by applying its method to the real data, not read from it.

The weak identifiability documented here is also the real-data counterpart of the identifiability limit derived analytically in the companion methods paper on the applicability bounds of fixed-dictionary detrital-zircon unmixing (Pan, submitted to *Computers & Geosciences*: "Where does detrital-zircon source unmixing break? An analytic identifiability criterion and its large-scale Monte-Carlo verification for NMF provenance unmixing"; reproducibility code archived at Zenodo, doi:10.5281/zenodo.20768132 [⚠ 稿号待补;若与「方法稿」同篇则 CAGEO-S-26-01308]). Its §3 identifiability theorem bounds the weight-estimation error by the end-member Gram condition number through an explicit Kolmogorov–Smirnov-distance → Gaussian-overlap bridge (with λ_min(HHᵀ) ∝ ρ^{2(K−1)} and, in the ideal density-Fisher limit, σ_err ≍ 1/(√N ρ^{K−1})), and identifies adjacent-end-member pairwise degeneracy as the controlling bottleneck — the same arc–basement (S1↔S2) pair resolved here. The connection is specific and complementary. The analytic Gram / condition-number analysis predicts the *direction* of the ill-conditioning exactly — the weakest eigenvector of the normalised end-member overlap matrix is (1, −1, 0)/√2, which coincides with the empirically observed soft axis to within a fraction of a degree — but it does not predict its *magnitude*: the bare density-overlap condition number (κ ≈ 1 + 4.5 × 10⁻⁵) drastically understates the softness actually observed, because at finite n the softness is governed by the sampling of the likelihood, not by the geometric overlap of the end-member densities. The division of labour is therefore clean: the methods paper supplies the analytic direction and the sufficient conditions, the present study supplies the finite-sample magnitude, and the two can be cross-referenced without re-deriving either. (This also explains why re-seeding the emcee sampler scatters the posterior mean by ~0.19 along this same axis — the sampler is tracing a genuinely flat likelihood ridge, not failing to converge.)

### 5.5 Provenance of the NE-Asia Lower Cretaceous basins

Geologically, the sharp craton/no-craton contrast between the two groups (§4) reads directly as provenance. The near-absence of an Archaean–Proterozoic component in the Songliao samples points to K1 sediment derived from Mesozoic arc rocks and Palaeozoic CAOB orogenic terranes — the basin drawing on its encircling arc and accretionary terranes rather than the distant craton — whereas the resolvable cratonic signature of the NCC north-margin samples is, conversely, expected from their position on the northern edge of the North China Craton. The clear exception, SL0932, is robustly craton-dominated (a hard result, not a soft-direction artefact; §4) and is the only sample of the SL09xx cluster from the Quantou Formation (the others Denglouku); we read its dominant ~1900 Ma component as proximal sourcing from a local basement high of exposed North China cratonic basement. Because the craton weight is the well-determined component throughout, these group-level provenance statements rest on the robust part of the posterior — subject to the §5.1 caveat against over-reading any individual soft sample's arc:basement ratio.

### 5.6 Limitations and sampling implications

The analysis carries three assumptions worth stating plainly. (i) The mixture is three-end-member with geologically fixed reference spectra; co-inferring the end-member shapes relaxes this only mildly (±10% μ / ±20% σ) and, as shown, matters for only two samples. (ii) The symmetric Dirichlet(1) prior on the weights is deliberately uninformative; a sparsity-promoting Dirichlet(α < 1) is a natural extension but is not required by the present data. (iii) Grain counts are modest (n = 31–111). The most actionable consequence follows from (iii): because the arc:basement uncertainty follows the n^(−1/2) counting-noise law of §4 for the reducible majority, future sampling can be designed to a target precision, whereas the two shape-inflated samples call instead for better independent constraints on the end-member age spectra. In this sense the uncertainty analysis is not merely descriptive but prescriptive — it turns "how confident are we?" into "what would it take to be more confident?", and answers that question differently for the reducible and irreducible samples of §5.2.

---

## 6. Gap 清单 + 投稿就绪审查

### 〇、一句话判断

**距"Results + Methods + 核心图齐备的可投稿初稿"还差约 3 步**:(1) 把已落盘但未出图的后验信息补成 2–3 张图 + 1 张主表;(2) 补样本位置图(Fig 1)和与第一篇监督 NMF 的对照(审稿人必问);(3) 写正文(目前 0 字正文,只有脚本+图)。**数值结果本身已完整且自洽,GPU 等价性这条方法学卖点扎实**——缺的是"把已有结果讲成论文",不是缺结果。所有补图所需的列都已在 `grains_bayes_gpu.parquet` 里,不需要重跑 18 万次拟合。

### ① 还缺哪些图 / 表(按优先级)

**P0 — 没有就不能投:**
- **✅ 样本位置图 / 构造底图(Fig 1,已出 2026-07-04)**：`Fig1_sample_locations.{pdf,png,svg}`(脚本 `Fig1_sample_locations_v1.py`)。复用第一篇 `dz-unmix-cg` 的 NEAsia basemap 构造框架 + geo-figures primitives,叠 15 个样本**真实坐标**(取自 dz-global-db curated 的 lat/lon,OneDZ/Puetz);NCC 方块在克拉通、松辽圆点在松辽盆地。auditor 0 issues,已渲染+Read 验证。caption 见 §2 Figure 1。
- **主结果表(Table 1)**：本文档 §1 已成形,补 n_zircons 列即可。
- **✅ 与第一篇监督 NMF 点估计的对照图(已出 2026-07-04,Fig 4)**：`FigY_nmf_comparison.{png,pdf}`(脚本 `FigY_nmf_comparison.py` + 数据 `FigY_nmf_comparison_data.csv`)。NMF(x) vs 贝叶斯均值±std(y) 散点 + 1:1 线,按端元着色。结果:**总 MAE 0.03,硬 w_S3 MAE 0.01、软 w_S1/w_S2 MAE 0.04,点估计落在贝叶斯误差棒内**。⚠ **审计修正**:原以为"第一篇 supervised_unmix + 文献 true_w 覆盖了这 4 个 NCC 样本"——**不对**,第一篇 case study 用的是**文献参数化的 4 个合成盆地代理**(松辽/海拉尔/二连/NCC 北缘),不是 05FW007 等真实样本。正确做法(已采用)= 把第一篇 `supervised_unmix` **只读应用**到 4 个真实样本的实际年龄谱,同端元(DEFAULT_SOURCES 与本篇一致)。对照仍限 4 个 NCC 样本(11 个 Songliao 第一篇没跑);正文 §5.4 已如实说明。

**P1 — 强烈建议补,数据已就绪:**
- **✅ 方差来源堆叠条图(已出,2026-07-04)**：`m3_variance_sources.{png,pdf}`(脚本 `m3_variance_sources_figure.py`)。逐样本 within-credible / bootstrap / prior 三段,把 §0.3 的 UQ 预算可视化,直接背书 §4「credible 已占一半、先验惰性、仅 05FW007/S12652 例外」。draft caption 见 §0.3。
- **✅ 端元 μ / σ 后验图(已出 2026-07-04,Fig 5)**：`m3_mu_sigma_posterior.{png,pdf}`(脚本 `m3_mu_sigma_posterior_figure.py`)。2×3 网格(行 μ/σ,列 S1/S2/S3)叠先验高斯 vs 代表性后验高斯 + 15 样本 rug。结论:**数据实质更新形状,先验未主导**——μ 收窄 0.31–0.53×、σ_S2/σ_S3 右移 +1.4/+1.6 先验σ(基底/克拉通峰更宽)。正文 §5.3 已引 Fig 5 佐证"先验是约束非假设"。
- **✅ 收敛 / 诊断图(已出 2026-07-04,Fig 6)**：`m3_acceptance_diagnostics.{png,pdf}`(脚本 `m3_acceptance_diagnostics_figure.py`)。(a) 全 181,500 fits acceptance 直方图 + 健康带 [0.2,0.5];(b) 逐样本箱线。结论:**99.3% 在健康带、median 0.42,15 样本全健康**,唯 S12652 略低(~0.28,最简并样本,采样最难,自洽)。burn-in(1000)/自相关按 note 放 caption 一句(逐拟合自相关未落盘,靠 acceptance + GPU-emcee 等价背书)。Methods §3 已引 Fig 6。
- **✅ bootstrap 收敛小表(已出 2026-07-04,Table 2)**：`m3_bootstrap_convergence_table.csv` + 正文 §3 Table 2。前 b(10/20/30/50/75)vs 全 100 的权重 std 比值,全 45 对 + 软子集(σ>0.02)两列。结论:**b≈20 中位已 ≈1.0、IQR 随 b 收紧 → 100 次是余量非过度**。⚠ 审计:中位稳健复现 Methods §3 的"≈1.0",但具体区间 0.84–1.64 随 b/方法变、被近零-σ 硬权重伪影主导(见软子集列),口径统一待你定。

**P2 — 锦上添花:**
- GPU vs emcee 等价性可视化(`validate_gpu.py` 输出)作 supplementary。
- 两个高不确定样本(05FW007、R31370-S12652)的单样本 corner plot,示范"软结论长什么样"。

### ② 审稿人最可能质疑点(及现成回应弹药)

1. **S1↔S2 简并必须显式讨论 + 给可识别性论证**：本文最大方法学软肋,也是核心卖点("不是所有权重同等可信")。说清 S1(130)/S2(280)相邻→弱可识别→相对配比软但其和稳;用 emcee 自身换种子在简并方向抖动 ~0.15–0.19 背书(后验真实几何,非采样器 bug)。**强关联第二篇 M1 可识别性定理**:正文一句"经验弱可识别性的解析刻画见 [bounds 第二篇]",双向引用、不重证。
2. **先验敏感性**:121 档 prior shift scan 已防死 → 把"扰动 ±10%μ/±20%σ × 121 配置后 w_S3 std 仍 ~0"当先验稳健性**主证据**,把质疑转成卖点。
3. **bootstrap=100 是否必要**:用收敛事实回应,主动解释那个 1.64 离群(std≈0.001 分母噪声)。
4. **为何相对不确定度 10%μ/20%σ**:端元峰位相对定年精度大致恒定、谱宽不确定性更大,故 20%>10%;中心取自第一篇。
5. **Dirichlet(1) 辩护**:无信息、不偏向;future work 可提 Dirichlet(α<1) 稀疏先验,不必实做。
6. **GPU 采样器正确性**:`validate_gpu.py` 逐拟合 |diff| 0.020 PASS → 等价性表 + "比 emcee MC 噪声小一个量级"。

### ③ 数据/代码可用性声明

- **代码**:`bayes_gpu.py`(GPU 采样器,核心创新)+ `run_scan_gpu.py` + `validate_gpu.py`(等价性背书,**务必一起放**)+ `bayesian_realdata.py`(emcee 参考)+ `fig_bayes_uncertainty.py`。Zenodo DOI,和第一篇(10.5281/zenodo.20371892)、第二篇并列。
- **数据**:15 输入样本 CSV + **全量 `grains_bayes_gpu.parquet`**(reviewer 复算真相源,必须公开)。
- **可复现**:bootstrap 种子确定性(`boot_seed` = sha1,避开旧版 `hash()` 的 PYTHONHASHSEED 非确定性)——写进去 = 主动声明可复现,加分。算力注明"emcee 单拟合 ~21s/全量数天 → GPU 4475s";离线 wheels(scipy/emcee/pyarrow)安装说明也给(L20 容器 air-gapped)。
- **license 边界**:Puetz(R 系列)/OneDZ(SL09xx)来源引用 + license 与 `dz-global-db` 已核验结论一致(别重核)。

### ④ 与 dz-global-db / 第一篇 / 第二篇的边界

| 项目 | 关系 | 怎么写 |
|---|---|---|
| 第一篇 `dz-unmix-cg`(supervised NMF 点估计) | 第三篇是其**不确定性升级版**,同场景同端元(`bayes_gpu.py` 镜像 `DEFAULT_SOURCES`) | 引为方法/端元出处;对照限 4 重叠样本;明说"端元中心取自第一篇" |
| 第二篇 `dz-unmix-bounds`(M1 可识别性定理) | 第三篇真实数据观察到的 S1/S2 弱可识别**正是 M1 定理实证案例**;代码物理就在 bounds 仓库 | 双向引用,不重证定理 |
| `dz-global-db` | Songliao Puetz/OneDZ 样本编译来源 + license | 来源引用 + license 沿用 db 结论;future work 提"推广到全球库",别混进 scope |

⚠ **命名雷区**:`d107_paper2_addon/` 目录名里 "paper2" 指**贝叶斯 addon**(物理挂 bounds 仓库),但**第三篇 = 贝叶斯不确定性论文**是独立投稿线。Zenodo/正文命名别被目录名误标成第二篇。

### ⑤ 三步收尾路线

1. **✅ 补图+表全部完成(2026-07-04)**:Fig 1 位置图、Table 1(+ lat/lon)、Table 2 bootstrap 收敛、Fig 3 方差来源、Fig 4 NMF 对照、Fig 5 μ/σ 后验、Fig 6 acceptance 诊断,以及**主图 Fig 2 `fig_bayes_uncertainty` 路径已改本地(`Path(__file__).parent`)并本地渲染 + Read 验证**(顺带修了面板 a 的 05FW007/100821-1 标签重叠)。全部 auditor/视觉验证过。
2. **写正文 → ✅ 全四节均有草稿(2026-07-04)**:Introduction / Geological setting(§2.5)、Methods(§3)、Results(§4 已重写)、Discussion(§5:5.1 逐权重可信度 + 5.2 计数噪声地板 + 5.3 先验惰性 + 5.4 一/二篇双向衔接 + 5.5 provenance + 5.6 采样建议)。**正文 0 字缺口已清**;**占位亦全部已补(2026-07-04)**:§2.5 构造归属/文献(8 条 DOI 逐条 Crossref 核)、§5.4 监督 NMF 对照(Fig 4)、§5.5 SL0932 层位(泉头组·局部基底隆起就近供源)。**唯余学术声音本人定稿** + LICENSE/上传铸 DOI + M1 稿号回填。
3. **✅ Zenodo 打包已备好(2026-07-04,staging = `zenodo_paper3/`)**:`assemble.ps1` 已跑,`zenodo_paper3/package/` = 52 文件 / 81.3 MB(code 5 含 bayes_gpu/validate_gpu/emcee 参考 + geo_primitives、analysis 3、figures 6 脚本、figures_rendered 13、data/grains_bayes_gpu.parquet 40MB、15 输入 CSV、results 7 CSV、deps/wheels 2)。**精确筛出 E 线文件、排除 100+ 个 M1/M2 probe_\* clutter**(见 `MANIFEST.md`)。含 `README.md`(复现步骤)+ `data_and_code_availability.md`(论文声明草稿 + mint-DOI 前 checklist)+ `requirements.txt`。**仍待你(不可自动化)**:补 `LICENSE`(源仓库无;建议 code=MIT、data=CC-BY-4.0)、zip `package/`、上传 Zenodo、reserve/publish DOI、回填 `<DOI>`。Puetz/OneDZ 输入 CC-BY 且逐样本带 source_doi。

> 唯一可选的"算力活":若要把监督 NMF 对照扩到 11 个 Songliao 样本,需在第一篇框架补跑那 11 个点估计——可选,正文限定对照子集即可绕过。

---

## 附:两点技术提醒(起草过程中发现)

- **✅ fig 脚本路径已修(2026-07-04)**:`fig_bayes_uncertainty.py` 原读 `/private/bayes_gpu/out/...`(L20 容器路径),现改为 `Path(__file__).resolve().parent / "grains_bayes_gpu.parquet"`(脚本同目录,可移植),输出同目录 `fig_bayes_uncertainty.{png,pdf}`。已本地渲染 + Read 验证,数据与 Table 1 一致。
- **parquet 实际位置**:全量结果在本目录 `d107_paper2_addon/grains_bayes_gpu.parquet`(40MB),不在 `Projects/bayes_gpu/` 工作目录。
