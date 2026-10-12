# 实证骨架:σ_err 标度律 + 简并相图(真实全球库) · v0.2

> **代号**:M1 §3 实证侧 — 用全球碎屑锆石库验证 §0.5a 主定理 σ_err=C/(√N·ρ₁ₖ),并给「推论(解析相图)」的**实证版**(此前 PROGRESS 状态总览里「未开始」)。
> **状态**:v0.2(2026-06-15,LN——§6 追加实测 NMF σ_err 直接标定 + K=3 floor 诚实更正)。v0.1(2026-06-14)本地原型,引擎在 `../../dz-global-db/`,结果均验过。**理论侧真相源** [`main_theorem_draft.md`](main_theorem_draft.md) §0.5a / [`PROGRESS.md`](PROGRESS.md);**桥到第三篇** [`../d107_paper2_addon/PAPER3_bayes_materials.md`](../d107_paper2_addon/PAPER3_bayes_materials.md)(贝叶斯实测 S1↔S2)。
> **数据源** = `dz-global-db` master_v6(1,788,250 颗,~9290 个 ≥60 颗的样本)。

---

## 0. 这份做什么 / 不做什么

**做**:把 §0.5a 主定理的两个可观测推论,在真实数据 + 受控合成上验出来——
1. **标度律** σ_err ∝ ρ⁻¹/√N(等价:可分性边界 ρ√n = const),
2. **简并相图**(K×ρ 失败面的实证版,= PROGRESS「推论」槽)。

**不做**(留作下一步,见 §5):**没有直接测 NMF 的 σ_err 本身**。本骨架用 **KS 两样本可分性**作 σ_err 的可观测代理(同族 ρ√n=const 几何),所以验的是**指数 + 几何**,不是前因子 C≈0.48。直接叠实测 NMF σ_err(`../d107_paper2_addon/sigma_err_scan.py`)是把 iso-σ_err 标签从"理论断言"变"实测"的那一步。

引擎(都在 `../../dz-global-db/`,配 `_*_report.txt`):
- `_proto_pairwise_gpu.py` — all-pairs 相异度矩阵的 GEMM/批量改写 + 对照 scipy + 计时(GPU 适配评估)
- `_proto_degeneracy.py` — bootstrap-CI + 同源噪声地板(对照 1.358√(2/n))
- `_proto_danger_boundary.py` — **成品图** `_proto_danger_boundary.png`(受控相图 + 真实定位)

---

## 1. 结果 1 — 标度律(验 §0.5a 指数)

受控:两高斯端元、分离度 ρ 解析已知(ρ=erf(Δμ/(2√2σ))),扫 (ρ, n) 平面测 P(KS 不可分)。
**经验 50% 不可分边界 ρ₅₀·√n = 1.35 ± 0.06(CV 4%,n=20→300)** ⟹ 实证 ρ₅₀ ∝ n^(−1/2),即可分性边界与 §0.5a 的 σ_err∝ρ⁻¹/√N **同族**(都是 ρ√n=const)。

- 这是 **pairwise(2 源)** 的律 —— 正是 §0.5a「σ_err 由极端/相邻 pairwise 对封顶」所指的那个绑定模,所以骨架直接打在主定理的瓶颈对象上。
- ⚠ 常数 1.35 是 **KS 50%-可分性常数**,**不是** σ_err 前因子 C≈0.48(后者是 NMF 权重误差的;两者同几何、不同常数)。C 的直接标定见 §5。

## 2. 结果 2 — 简并相图(填 PROGRESS「推论」槽)

成品图**面板 A**:P(不可分) 热图(magma)+ 经验等值线(P=0.5 / 0.9)+ M1 iso-σ_err 参考线(C/(σ·√n))。肉眼可见**经验简并等值线 ∥ iso-σ_err 线**——这就是「K×ρ 失败面」的实证相图(此前未开始)。其解析版仍为 TODO。

## 3. 结果 3 — 真实全球库定位

`_proto_danger_boundary.py` 面板 B(N=300 池,44,850 对):
- 中位 ρ̂ = 0.72;**随机全球对几乎不简并**:稳健简并 **0.00%**、模糊 **3.08%**(筛 margin<0.1 尾 → 只 bootstrap 这 5.7%)。
- 简并是 **(小 ρ + 小 n) 角**的条件现象 —— 随机全球抽样碰不到,但**少颗粒的区域研究正踩这里**。
- 物理含义:全球 DZ 物源空间**绝大多数良态**,M1 的高 K 灾难"观测不到"与此一致(§0.5a 物理结论)。

## 4. 结果 4 — S1↔S2 对账(桥到第三篇贝叶斯)⭐

项目绑定对 **S1(arc 130±15)↔S2(CAOB 280±30) 的 ρ = 0.999**(KS,几乎完全可分)。

与第三篇贝叶斯实测对账(`PAPER3_bayes_materials.md`:"w_S1↔w_S2 weakly identifiable owing to end-member degeneracy",Songliao 沿 S1–S2 边拉长):
- **不矛盾,且收紧措辞**:S1↔S2 是三对里**最近的绑定对**(相对最软方向,= §0.5a 物理结论的"相邻 pairwise"),贝叶斯后验沿这条边展宽是对的;
- 但绝对上 ρ=0.999 **远在危险边界之外** ⟹ 那个后验展宽**主要由有限颗粒数(小 N,实测 n=31–111)驱动,不是端元简并**。
- ⟹ 建议第三篇把"owing to end-member degeneracy"改成更准的"**the relatively softest (binding) direction, with the spread dominated by finite grain counts rather than end-member overlap (ρ_S1S2≈1.0)**"。这条对账是**两篇统一**的具体落点。

## 5. caveat + 下一步

**caveat(写进正文前)**:
1. **KS 可分性是 σ_err 的代理**,非 NMF 权重误差本身 → §1 验的是指数/几何,C 未标定。
2. 面板 B 的 x 轴 ρ̂ 对小分离对有**正偏**(KS 是 sup,有限样本高估真 ρ)→ 危险被**低估**。
3. 面板 B 虚线边界来自受控律外推、点色来自各对 bootstrap,来路不同,正式图须标注。

**下一步(把代理升级成直接证据)**:跑 `../d107_paper2_addon/sigma_err_scan.py`(本地 numpy 后端,K=2/3),把**实测 NMF σ_err** 叠到同一张 (ρ,n) 平面 —— 坐实"KS 测不出区别的边界 = unmixing 误差真正爆掉的地方",iso-σ_err 标签从理论断言变实测,一张图缝死 bounds §3 与第三篇。**✅ 已完成(2026-06-15),见 §6。**

**算力**:全程本地秒级;全量 all-pairs×bootstrap 才需 L20(~6h),目前不必要。

---

## 6. 直接证据:实测 NMF σ_err 叠相图(§5「下一步」落地)· 2026-06-15 LN

把 §5「下一步」做了:用 [`../d107_paper2_addon/sigma_err_scan.py`](../d107_paper2_addon/sigma_err_scan.py)(numpy 后端 = 调 real `dz-unmix-cg` 核,R=600,**endpoint w**——按 PROGRESS 实验设计,endpoint k\* 才是干净 pairwise σ_err)在 **(ρ,n) 平面直接测 supervised-KL-NMF 的 σ_err**(128 cell:K=2/3 × ρ∈[0.05,0.47] 8 档 × n∈[20,350] 8 档)。成品图 [`../d107_paper2_addon/figures/fig_sigma_err_phase_v1.png`](../d107_paper2_addon/figures/fig_sigma_err_phase_v1.png),脚本 `../d107_paper2_addon/fig_sigma_err_phase.py`,数据 `../d107_paper2_addon/out_phase/sigma_err_slope_numpy.json`。**把 §2 面板 A 里理论公式画的 iso-σ_err 线换成了实测等值线 → §5 caveat 1「C 未标定」解除。**

**结果 1 — K=2 pairwise 律被 NMF 直接标定(不再靠 KS 代理)**:
- LAN cell 实测 **C = σ_err·√N·ρ = 0.500 ± 0.026(CV 4%,29 cell)** ⟹ 命中 [`pairwise_threshold_calibration.md`](pairwise_threshold_calibration.md) 的 **C≈0.48**。
- σ_err vs 1/(√N·ρ) 的 log-log 斜率 **+0.95(理论 +1)** ⟹ σ_err ∝ ρ⁻¹ 指数实测确认。
- 这是 §1 用 KS 可分性代理验「指数+几何」之后,**直接测权重误差本身**把前因子 C 也钉死的那一步。

**结果 2 — 一句话缝合(面板 A)**:实测 iso-σ_err 等值线 ∥ KS P=0.5 简并边界 ∥ 理论 ρ√n=const,三者同族双曲线。定量落点:
> **KS 不可分边界 ρ√n=1.35 ≡ 实测 NMF σ_err ≈ C/1.35 ≈ 0.37 的等值线。**

即「KS 测不出端元区别」= 「权重误差爆到 ~0.37」,bounds §3 ↔ 第三篇在同一双曲线上焊死。

**结果 3 — ⚠ K=3 高于 K=2 floor 的真机制 = curvature 模主导(诚实更正 ×2)**:
- 初版图曾标「K=2/3 K-无关坍缩」**已撤**;**本节首版(同日)把机制写成「finite-n 曲率未冻结、需 N≫ρ⁻⁴、大 N 会消失」——经特征向量诊断证伪,一并更正。**实测:K=3 在 n≤350 下 σ_err **高于 K=2 floor ~35-49%**(不是初稿的 25%;ρ=0.35-0.47 区 excess=1.35-1.49)。
- **诊断**([`../d107_paper2_addon/diag_k3_mode.py`](../d107_paper2_addon/diag_k3_mode.py),ŵ 协方差**特征向量**,R=600 real 核):单纯形切空间 2 维,两正交模 u1=(1,0,−1) pairwise / u2=(1,−2,1) curvature。
  - **大 ρ(>ρ_c≈0.32,unpinned)**:**worst 模 = curvature u2**(top 特征向量 |·u2|≈1.00、|·u1|≈0.05)。曲率模幅度随 ρ 仍**下降**(实测 σ_u2 的 ρ-slope ≈ −1.2(N=350)→ −1.7(N=1500),比 ρ⁻¹ 略陡;N-slope ≈ −0.47),但**前因子 C_curv=σ_u2·√N·ρ 不是常数**:unpinned 窗内随 ρ 单调降,**C_curv≈0.70–0.85**(ρ≈0.47 端 ~0.70、ρ≈0.34–0.40 端 ~0.78–0.86)。相对 K=2 floor(C≈0.50)的 excess=**1.35–1.49×**(结构性,n≥70 plateau,不随 n 消失;w 依赖弱,近顶点 w 降到 ~0.45–0.59 但模仍是 curvature)。
  - **小 ρ(<ρ_c,pinned)**:**curvature 冻结**(C_u2→0.03,|·u1|≈1.00),worst 退回 extreme-pairwise C/(√N·ρ₁ₖ),σ_err **低于** naive floor(excess<1)——正是 [`curvature_freezing_proof.md`](curvature_freezing_proof.md) 的 active-set 钉扎,**此处独立复验**。
- **正确图景(两模 max)**:σ_err = max(pairwise-extreme, curvature)。**1/√N 半部全程 K-无关(实测 N-slope ≈ −0.5)✓**;**ρ 依赖是 regime-依赖的,不是全程 ρ⁻¹**:小 ρ 钉扎区 = C/(√N·ρ₁ₖ)(curvature 冻结);大 ρ unpinned 区 curvature 活,σ_err 随 ρ 降(slope ≈ −1.2…−1.7),前因子 C_curv≈0.70–0.85(非常数)。**σ_err 跨 ρ 非单调、在 ρ_c≈0.32 邻域达峰**——跨 ρ_c 的朴素 log-log 拟合给虚假 ~0/+斜率(regime-mixing artifact,**勿据此说「curvature 区 ρ-flat」**——对抗自检里有 lens 一度据此误判,被综合裁定纠回)。crossover ρ_c=erf(bw/(2√2σ√D)),endpoint w* 的 D=0.333 ⟹ **ρ_c=0.322**(实测 excess=1 越界 ρ≈0.28、eigvec flip ρ≈0.245)。
- ⟹ **⚠ 对 §0.5a 正文的潜在修正(待用户数学判断,未擅改正文)**:§0.5a 头条「σ_err=C/(√N·ρ₁ₖ),K-无关」**只描述小 ρ 钉扎区**;大 ρ 由 curvature 模主导,C_curv≈0.70–0.85 比 ρ₁ₖ 公式预测(~0.30)高 ~2.3×。完整应写两模 max。「K 鲁棒」对 **1/√N 指数**成立、对 **ρ-依赖与前因子**不成立。**注意**:大 ρ 区不是干净 ρ⁻¹(ρ-slope −1.2…−1.7),σ_err 在 ρ_c≈0.32 附近非单调达峰;「C_curv/(√N·ρ)」仅是 ρ≈0.47 邻域的近似拟合,C_curv 非常数。
- **验证状态(2026-06-15,6-agent 对抗自检已过)**:5 lens(seed/R 稳定、NMF 收敛、true-w 依赖、与 GPU/标定理论自洽、ρ_c 边界)+ 综合裁定。**幸存**:curvature=大ρ worst 模、结构性 ~1.4× excess、小ρ 冻结、1/√N K-无关、§0.5a 仅覆盖钉扎区。**被证伪并已修正**:(a)「全程 ρ⁻¹」——大ρ区 ρ-slope −1.2…−1.7 且 σ_err 非单调;(b)「C_curv=常数 0.70」→ 带 0.70–0.85;(c)「ρ_c≈0.2」→ 0.32。综合裁定还纠正了 theory_consistency lens 自身的错误修复(它说「ρ-flat」是跨 ρ_c 拟合假象)。脚本 `../d107_paper2_addon/{diag_seed_stability,conv_probe,conv_probe2,diag_w_sweep,judge_rho_exponent}.py`。
- **K≥4 扩展（2026-06-15，第 2 轮 6-agent 对抗自检过）**：worst 模 = **degree-(K−1) contrast 多项式**（宇称 (−1)^{K−1}；K=3 曲率 (1,−2,1)=degree-2 特例；K=2..5 top 特征向量与 degree-(K−1) overlap **0.98–1.00**，w/bw-稳）。excess 随 K **对数饱和** **a+b·ln(K−1)**（R² 0.93–0.998；ρ=0.47 K=2..7=1.04/1.43/1.62/1.84/1.92/1.94，K≈6 趋平；ρ=0.40 K=7 回落，收敛守卫确认非欠收敛），**非无界单调**；excess 依赖 ρ。**机制更正**：worst=degree-(K−1) **不是** KDE 低通（bw 大→excess 小、σ_err~bw^{−0.5}、H 由 σ 构造 bw-无关 bit-identical）；解析成因 open。详见 [`main_theorem_draft.md`](main_theorem_draft.md) §0.5a (II) v0.4。脚本 `diag_kge4_{mode,poly}.py` + workflow `probe_*`。

**caveat 状态更新(对 §5)**:caveat 1(C 未标定)→ **已解除**(C=0.500 直接测得);caveat 2/3 仍适用于面板 B 的真实对。

**算力**:全程本地 ~10 min(128 cell × R600,real 核全收敛 n_iter=2000);深 LAN K-无关坍缩 + K=5 仍须 L20(已就绪 `Projects/sigma_err_gpu/`)。

memory:[[dz-degeneracy-empirical-backbone]] / [[m1-identifiability-theorem-line]]。
