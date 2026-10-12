# excess(K) 解析推导 · 工作日志（深推导,2026-06-16 起）

> **目标**:解析推导 §0.5a (II) 的 unpinned 区 excess(K,ρ) —— ln(K−1) 系数 + ρ-依赖 + K≈7 回落。
> **状态**:✅ **完成(2026-06-16)**。Phase 0(框架+分解)→ Phase 1(λ_min/S 渐近:p∝1/erfinv² + ω_q)→ Phase 2(clipping 回落 + 对抗自检 + 第一性原理)→ 已并入 §0.5a (II)。结论见末「最终结论」节。
> **真相源**:本日志 = 推导过程;结论并入 [`main_theorem_draft.md`](main_theorem_draft.md) §0.5a (II)。脚本 `../d107_paper2_addon/{compute_sandwich,compute_factors,probe_turnover,analytic_worst_mode}.py`。

---

## Phase 0 — 框架 + 分解（已立,数值验证）

### 0.1 M-估计 sandwich 框架（`compute_sandwich.py`,复现实测 excess ~10%、log-fit R²=0.994）
σ_err² = λ_max[ G⁻¹ Σ G⁻¹ ]|切空间,
- **G_ij = ∫ h_i h_j / q**（信号 Fisher = KL Hessian;最小特征向量 = degree-(K−1) contrast,已证）
- **Σ_ij = (1/n) Cov_{a~q}[φ_i, φ_j]**,φ_i = (h_i/q) ⋆ K_bw（KDE 平滑 score kernel）
- bw→0:Σ|切空间 = G/n ⟹ σ_err²=1/(n λ_min)（Cramér-Rao 界）。

### 0.2 干净分解（`compute_factors.py`）:**excess(K) = CR(K) · S(K)**
- **CR(K) = √(λ_min(2)/λ_min(K))** —— 信号 Fisher 包络;λ_min ~ (K−1)^{−p(ρ)}。
- **S(K) = √(Var_q[g_v⋆K_bw]/Var_q[g_v])** —— bw 抑制(g_v=Σv_i h_i/q,v=worst 模)。
- **关键**:p(ρ) **ρ-依赖**:
  - ρ=0.47:p≈0.95 ⟹ CR~(K−1)^{0.48}~√(K−1);S:0.96→0.60(K=2→10)。
  - ρ=0.40:p≈1.88 ⟹ CR~(K−1)^{0.94}~(K−1);S:0.97→0.56。
- 积 CR·S 在 K=2..7 拟合 **a+b·ln(K−1)** 极好(ρ=0.47:a=0.95/b=0.44/R²=0.994;ρ=0.40:a=0.78/b=1.14)。
- **⟹「ln 律」是幂律×缓降的局部表象,不是基本对数;b 由 p(ρ) 决定**(这解释 ρ-依赖)。
- ⚠ 谜:g_v 的 rms 频率 ω_eff ~ K-无关(~0.05–0.07),但 S 缓降 → S 不是简单 exp(−bw²ω²);真正驱动 S 的谱特征待 Phase 1 厘清。

### 0.3 回落 = 非负 clipping（`probe_turnover.py`,ρ=0.40,n=350,R=800）
| K | meas/sand | frac(ŵ<0.01) | w*_min |
|---|---|---|---|
| 2–4 | 1.00–1.05 | 0.000–0.009 | 0.19–0.38 |
| 5 | 0.89 | 0.063 | 0.15 |
| 6 | 0.88 | 0.061 | 0.125 |
| 7 | 0.69 | 0.099 | 0.107 |
| 8 | (—) | 0.096 | 0.094 |
- 边界命中率在 **K=5 突抬**(0.009→0.063),**正好** = measured 跌破 unconstrained sandwich 处。
- 机制:endpoint w 的 **w*_min ~ O(1/K) 随 K 缩 + σ_err 涨** ⟹ 高 K 时 worst-mode 涨落把最小权重推到 0、**截断方差** ⟹ 回落。
- **回落模型(待 Phase 2 量化)**:excess 沿 unconstrained CR·S 走,直到 **clipping onset ≈ 当 σ_err ~ w*_min**(ρ=0.40 时 K*≈5);之后饱和/回落。
- ρ=0.47:clipping 弱(σ_err 小),measured 略高于 sandwich(在 ~10% 框架误差内),无强回落。

---

## Phase 1 — λ_min(K,ρ) 与 S(K) 渐近（workflow `w1s4eqij1`,6 agent,已完成 2026-06-16）

### 1.1 λ_min(G,K,ρ)（3 法一致,Toeplitz 符号最干净）
- **重叠矩阵 M_ij=∫h_i h_j 精确 Toeplitz**（验到 2e-11）:m_k = c·exp(−b k²),c=1/(2σ√π),**b = Δ²/(4σ²) = 2·erfinv(ρ)²**（精确恒等式,Δ=2√2·erfinv(ρ)·σ）。
- 符号 Poisson 求和 f_M(θ)=(1/(2σ√b))Σ_n exp(−(θ−2πn)²/(4b));worst 模在最高频 θ_K≈π(K−1)/K,λ_min(M)~f_M(θ_K)~exp(−θ_K²/(4b)) 近 θ=π。
- **worst-mode Rayleigh 恒等式**:vᵀGv/vᵀMv = ∫g²/q / ∫g² = ⟨1/q⟩_g（**精确 0% 误差**）⟹ λ_min(G)=λ_min(M)·⟨1/q⟩,p_G=p_M−p_R。
- **核心 ρ-依赖（严格、三法一致）:p ∝ 1/erfinv(ρ)²** —— 来自符号衰减常数 b=2erfinv(ρ)² 以 π²/(4b)=π²/(8erfinv²) 进入近 θ=π 的高斯符号。
- 综合裁定的 p(ρ) 拟合（**窗口依赖**,因 λ_min 是 theta-函数下降、非真幂律）:
  - K=2..7 窗（实测 0.95/1.88 来自此窗）:**p(ρ) = 0.241/erfinv(ρ)² − 0.194**（R²=0.98）。
  - K=2..10 窗:p(ρ) = 0.295/erfinv(ρ)² − 0.531。
- ⚠ **诚实表述**:λ_min 不是真幂律(theta 下降),"指数 p" 是局部 log-斜率;**ρ-不变量是 p∝1/erfinv(ρ)²**,具体数值随窗口。p_R≈0.37 仍经验(1/K 平台预测 ~1、被 pairwise 重叠封到 3/8,第一性原理 open)。

### 1.2 S(K,ρ,bw)（2 法等价,omega_eff 谜解开）
- **精确闭式**:S² = [vᵀG_s v − (vᵀb)²]/(vᵀGv),G_s,ij=∫q φ_iφ_j,φ_i=(h_i/q)⋆K_bw（= Σ 核的 bw 衰减;**<0.1% 匹配**）。
- **omega_eff 谜 RESOLVED**:控制量不是无权 FFT 频率（因 g_v=(Σv_i h_i)/q 在 q→0 尾部爆炸、主导 Lebesgue 谱但 Var_q≈0）,而是 **q-加权 Rayleigh 频率 omega_q² = E_q[g'²]/E_q[(g−⟨g⟩)²] = (vᵀAv)/λ_min**,A_ij=∫d_i d_j/q,d_i=h_i'−h_i q'/q。omega_q 随 K 涨（2.46×/2.63× over K=2..10）正好跟踪 S 下降。
- **worst-G 模 = 铅笔 (A,G) 的最大广义特征向量**（overlap 0.997–0.999）⟹ S 降因 omega_q²=(弯曲能)/λ_min 随 λ_min 崩塌而升 —— **与 (1.1) 的 λ_min 同一个**。
- 闭式精度阶梯:单色 S=exp(−½bw²omega_q²) RMS 4–8%（Jensen 负偏）;二阶矩 3–6%;**仅 KxK 精确式 <0.1%**。bw=12 不是小量（bw·omega_q~1）。
- **S 本质是伸展指数（−lnS=α(K−1)^β,β≈0.9–1.0）,非幂律。** omega_inf(ρ)≈0.92·Nyq/√κ、a≈2.4/κ²（κ=2erfinv²）经验、第一性原理 open。

### 1.3 合成 excess=CR·S + 「ln 律」的真相 ⭐
- **「a+b·ln(K−1)」是窗口巧合**:幂律 CR=(K−1)^{p/2} × 伸展指数 S 在 K=2..7 上像对数,**不是基本对数律**。论文须诚实写。
- **b(ρ) 解析**:**b(ρ) = 0.679·(p(ρ)/2)·⟨excess⟩₂..₇**（R²=0.9987）——精确复现 b(0.47)=0.44、b(0.40)=1.14。b 随 ρ 降而升,因 p（Fisher 崩更陡）和 ⟨excess⟩（幅度更大）双升。0.679=(p/2−q_S)/(p/2) 的 S-抑制 deflation（q_S≈0.24 ρ-flat,第一性原理 open）。
- **无约束饱和峰**:K=8(ρ=0.47)/K=12(ρ=0.40)/>14(ρ=0.35)——**无约束分析不产生 K~7 回落** ⟹ 实测 K~7 回落**必为非负 clipping**（= Phase 0.3 经验靶,Phase 2 建模）。

### 1.4 Phase 2 待办（裁定列出）
1. **K~7 回落 = 非负 clipping 建模**（中心项;无约束 Fisher 无边界,Phase 0.3 已测 onset K≈5@ρ=0.40）。
2. 第一性原理 p_R≈0.37。3. 第一性原理 omega_inf(ρ)、a(ρ)（需边界修正 banded-Toeplitz 边缘本征）。4. b(ρ) 的 0.679 deflation 系数。
- 脚本:`probe_reconcile{,2,3}.py` + 各 attempt 的 `probe_*.py`。

---

## Phase 2 — 合成 + 回落模型 + 对抗自检（待）

合成 excess=CR·S 的闭式、b(ρ)=f(p(ρ)) 的解析、clipping 回落 K*(ρ) 模型;然后对抗自检;并入 §0.5a (II)。**[待]**

---

## Phase 2a — 三个经验前因子的第一性原理尝试（"firstprinc",2026-06-16）

> 脚本:`../d107_paper2_addon/probe2_pR_*.py`、`probe2_omega_*.py`、`probe2_deflation*.py`。
> 所有结论已对 real-core(`sigma_err_scan.peak_ages`+`fixed_interior_w`)交叉验证(<1/q>_g 数列逐位一致)。

### 2a.1 p_R(≈0.37):**机制找到、cap 部分闭合,但不是普适 3/8** —— NEEDS-REVISION
- `<1/q>_g = (vGv)/(vMv) = 1/Qeff`;measured p_R(K=2..7 窗)= 0.47/0.39/0.33/0.28(ρ=0.47/.40/.35/.30,**随 ρ 升**),"0.37" 仅 ρ≈0.40 的窗口值。
- **不是 weight-imbalance 假象**:flat w 与 endpoint w 给同一 p_R(0.524 vs 0.518@ρ=0.47)——纯几何。
- **"1/K 平台" 真相**:mode-agnostic flat `<1/q>` 指数(0.65@ρ=0.47,[K])确实更大、由 q 的 1/K 背景稀释驱动(beta0=∫h²/q ~ K^{0.77});worst-mode 加权把它 deflate 到 p_R。
- **3/8 cap 机制 = worst-mode 对角存活分数 D=W0/Σ|W_k|**:**W0=c 精确 K-无关**(=Σv_i²M_ii=m0);D 随 K 降向极限。**关键:worst mode = 带状 Toeplitz 的 Dirichlet 顶 sine 模 v_n=sin(nπK/(K+1))**——sine-mode D 复现 true D <0.5%(ρ=0.47 K=16:0.3617 vs 0.3635)。
- **但 D_inf(1/K 外推)随 ρ 变**:0.324/0.266/0.229/0.197,**非普适 3/8**。"3/8" 只是 ρ≈0.47、K≈7–12 窗的 D 值。
- **障碍**:p_R = D_inf·p0 不闭合(0.18 vs 0.39):`<1/q>_g` 是 beta0(陡)与 beta_{k≥1}(缓且增长)的 D-加权混合,不能因式分解成 cap×dilution。且 λ_min 是 theta 下降→p_R 本就是窗口局部斜率。**诚实表述:cap 的解析对象是 sine-mode 的 W0/Σ|Wk|(可算),但 p_R 无干净 ρ-不变闭式。**

### 2a.2 omega_q(ρ)、a(ρ):**(K-1)/(K+a) 形式 = Dirichlet 边缘本征,验证极好** —— SURVIVES
- **omega_q² = omega_inf²·(K-1)/(K+a) 拟合 R²=0.996–0.9998**(K=2..12)。这正是带状 Toeplitz Dirichlet 边缘本征的线性分式(Möbius)签名:x=π/(K+1) 的边缘 gap,omega_q²=(a0+a2 x²)/(g0+g2 x²),K→∞ 趋 a0/g0=omega_inf²(x²-展开拟合 R²>0.9996,独立确认边缘结构)。
- **omega_inf = 0.92·Nyq/√κ 验证**(Nyq=π/dmu,κ=2erfinv²):fit/pred = 0.88/0.84/0.86/0.97(均值~0.89,**0.92 系数支持到 ±10%**)。
- **a = 2.4/κ² 验证**:fit a/(2.4/κ²)=0.66/0.67/0.75/0.99;**函数形式 a∝1/κ² 支持**,系数 2.4 偏高,2.0–2.4/κ² 更佳(残差因 λ_min theta 非幂律使窗口常数漂移)。

### 2a.3 0.679 deflation:**q_S≈0.24 真实且 ρ-flat;但 0.679 是跨-ρ 全局回归系数,非干净 (p/2−q_S)/(p/2)** —— NEEDS-REVISION
- **精确斜率恒等式**:b = window-mean d(excess)/dln(K-1) = `<excess·(p/2 − s_S)>`,s_S = −dlnS/dln(K-1)(S 的局部 log-斜率)。⟹ deflation = 1 − <excess·s_S>/((p/2)<excess>)。
- **q_S ≈ 0.24 强确认且 ρ-flat**:q_S(fit)=0.240/0.261/0.265/0.242/0.191(ρ=0.30..0.55,均值 0.240,std 0.026)。这部分 claim 成立——S 的有效 log-斜率是 ρ-不变的 ~0.24。
- **0.679 = 全局 leverage-加权回归系数**(由小 ρ/大 b 点主导):用 reconcile3 完全相同定义(p7=窗口 λ_min 斜率)**精确复现 K0=0.679,R²=0.9987**;per-ρ K0=b/((p7/2)<exc>)=0.681/0.684/0.662/0.579/0.331——**在 ρ=0.30–0.40 平在 ~0.68,但 ρ≥0.47 下滑**,非普适常数。
- **障碍**:(p/2−q_S)/(p/2) 不能跨 ρ 恒定,因 p/2 随 ρ 强变(0.5→1.5)而 q_S flat(~0.24)→比值必随 ρ 变。per-ρ K0 在小 ρ 近似稳是因 <excess·s_S>/<excess> 部分跟踪 p(excess 在 p 大处更陡),非因该代数式恒定。**诚实表述:q_S~0.24 是真前因子;0.679 是全局拟合数,不是 (p/2−q_S)/(p/2) 的解析值。**

---

## Phase 2b — K~7 回落 = effective-K / active-set 约简（"clip_effectiveK",2026-06-16）

> 脚本:`../d107_paper2_addon/probe2_clip_effectiveK*.py`、`probe2_clip_{mechanism,eigcross,mixture,truncation,effK_reconcile}.py`。
> 与并发 agent 的单壁截断高斯模型(`probe2_clip_{truncgauss,model,final,verify}.py`)交叉核对。
> 所有结论对 real-core(`sigma_err_scan` endpoint w, ρ=0.40/0.47, n=350)交叉验证。

### 2b.1 回落是真实的(非 R-噪声)—— 高 R 自举确认
- `probe2_clip_eigcross.py`(R=4000,自举 sd):exc(K=6)−exc(K=7)=**0.297 = 11.1σ**。K*(measured)=6 稳健。
- ρ=0.47 同样 K*=6(1.82→1.77,浅回落);**两个 ρ 在 n=350 都 K*=6**——K* 由 endpoint 几何(w*_min)+ 绝对 σ_err 标度定,**ρ 主要调回落深度不调位置**。(注:并发模型早期"ρ=0.47→K*=7"是 hand-typed SAND dict 截到 K=7 的外推假象,见 2b.4。)

### 2b.2 机制 = 顶本征值 CROSSOVER（active-set 约简的本征值表述）⭐
- worst 本征向量的 contrast 度数 deg1 在 K=2..6 精确跟 K−1;**K=7 时 lam1 下降(0.0266→0.0203)而 lam2 继续升**——clipping 把最高度数(K−1)模截断到塌过次模(K−2)。
- ⟹ 测得 worst 模 = effective-(K−1) 系统的 worst 模。**"丢一个 active endmember" = deg 从 K−1 掉到 K−2**,这正是 active-set 约简,在本征值层面就是 crossover。
- 这是任务要求的 effective-K 图像的**正确严格表述**(naive"按 frac 加权 drop-one"图像见 2b.3 失败)。

### 2b.3 naive effective-K 混合 FAILS,正确的是 boundary-proximity 截断
- naive 双态:excess_clip² = (1−f)exc_unc(K)² + f·exc_unc(K−1)²,f = probe_turnover 的 frac<.01。**过预测**(K=7 R_pred=0.99 vs R_meas=0.69)。两个原因(`probe2_clip_mechanism`):
  1. **f 用错了**:frac<.01 是 per-ENTRY 分数(R×K 个),真正的 per-REALIZATION 命中率 p_clip(K=7)=**0.62** vs frac_ent=0.12(欠数 ~5×)。
  2. **exc_unc(K−j) 用错了**:即使 no-clip(j=0)子系综 excess 也被压制且**回落**(exc_cond(0):1.70→1.76→1.59→1.40 over K=5..8)——边界邻近就把最高度数模的高斯涨落单侧截断,不必硬命中 0。
- `probe2_clip_effK_reconcile`:全方差恒等式 σ_clip²=Σ_j P(n_clip=j)σ_cond(K,j)² 下,exc_cond(j) 非干净单调降(K=6,8 处 clipped 子系综因条件均值漂移反而更大,between-state 项)。**诚实表述:active-set 约简对,但其强度来自边界邻近截断顶模(=crossover),单壁高斯低估。**

### 2b.4 K*(ρ) 闭式 + 两模型一致性
- **闭式 onset 条件:绝对 σ_err 顶模 ≈ w*_min(K)**,endpoint w 下 **w*_min(K) = 0.75/K 精确**(=0.6/(0.8K))。
  - ratio σ_abs(K)/w*_min(K) 过 1:ρ=0.40 在 K=4(0.70)→K=5(1.09) ⟹ onset K≈5;**深回落 K*=6(onset 后 ~1 步,clipping 充分展开)**。
  - ρ=0.47 ratio 在 K=7(1.01)过 1 ⟹ onset 晚、clipping 弱、浅回落,K*=6 恰在 ratio→1 处。
- **两个 clipping 模型在每个 ρ 一致**(`probe2_clip_effectiveK_crossover` 交叉核对,|ΔK*|=0 跨 ρ=0.35–0.50):单壁截断高斯(clip 顶模坐标)与 effective-K crossover(顶模塌过次模 = (K−1)-active worst 模)是**同一事件的两种分辨率**。
- **clip-hardness κ=1.805**(并发 agent 在 ρ=0.40,K=7 单点冻结):真实 KL-NMF 的迭代 MUR 比单次高斯投影 clip 更硬(active-set 复合),κ=1 是下界。crossover 模型沿用同一冻结 κ。
- **数值匹配**:effective-K crossover 模型 exc_clip vs measured(R=4000)mean|resid|=10.3%;per-K 约简 meas/sand mean|resid|=5.9%。单壁模型对 meas/sand(probe_turnover)mean|resid|<9%(`probe2_clip_final`,K≤7)。

### 2b.5 裁定与诚实边界
- **SURVIVES**:回落机制 = 非负 clipping 的 effective-K/active-set 约简,本征值表述 = deg-(K−1)→(K−2) crossover;K*(n=350,ρ=0.40/0.47)=6,11.1σ 真实;onset 闭式 σ_abs(K)≈w*_min=0.75/K。
- **NEEDS-REVISION / 诚实表述**:
  - κ=1.805 是单点经验冻结(MUR 硬度第一性原理 open);
  - 单壁高斯模型未截断时(用真实非截断 unconstrained 曲线)K* 漂到 ~10——**它的"K*=6"依赖经验 κ + unconstrained 曲线在 ~K=11 饱和的配合**,不是纯几何必然;
  - exc_cond(j) 的 between-state 项使简单"drop-j 混合"不闭合,真实强度需边界邻近截断(crossover),无单参数干净闭式给 exc_clip 全曲线。

---

## Phase 2c — 对抗自检 + 单壁 clipping 模型（2026-06-16）

### 2c.1 clip 单壁截断高斯模型（"clip_truncgauss",= 2b crossover 的等价单壁视角）
- ⚠ **纠正**:naive"最小 endpoint 权重 clip"**假**(那些 |v_k| 小、从不命中 0)。**真相:clip 绑在与 worst 模 v 最对齐的内部端元**(大 |v_k|、被正向涨落压下)。worst-模坐标 a=v·(ŵ−w*) 在 **K=5 起大正偏度**(0.04/0.06/−0.05|+0.67/+0.15/+0.75)= 单边截断签名。
- 模型:a~N(0,σ_unc²),单边墙 a_wall=min_{k:v_k>0} w*_k/v_k,Λ=a_wall/σ_unc,**meas/sand=[g(Λ_K)/g(Λ_2)]^κ**,g(L)=Var[min(X,L)]、κ=1.805。Λ(ρ=0.40)=7.70/4.13/2.78/1.89/1.49/1.20 → K=5 穿 ~1.9 = onset。
- 数值:meas/sand 模型 1.00/1.00/0.99/0.91/0.80/0.69 vs 实测 1.00/1.01/1.05/0.89/0.88/0.69(平均|err|~3%);独立 MC(高斯抽样+单纯形投影,κ=1)同 onset/方向。脚本 `probe2_clip_{truncgauss,which,model,mc,final,verify}.py`。

### 2c.2 对抗自检（2 lens）
- **「ln 是窗口巧合」: SURVIVES 强**。跨 4 个 K-窗 a+b·ln(K−1) 系数漂移 b 34–92%、a 至 1328%(变负);CR 实为幂律(R²_pow≤0.996)、S 实为 stretched-exp(β=0.95–1.16)。"ln(K−1)" = (幂律)×(stretched-exp) 在 K~2..7 的拟合假象。
- **「p ∝ 1/erfinv(ρ)²」: NEEDS-REVISION**。仅是**渐近律**(large-K/小-ρ、θ→π 符号;top-K 局部斜率 →2 = 1/erfinv²);**有限窗 K=2..7 假**(p 被压扁到 ~ρ^{−0.8},仿射 1/ρ 拟合更好 R² 0.998 vs 0.978)。正文须作渐近不变量陈述 + theta-下降 caveat。
- **「b(ρ)=k·(p/2)·⟨excess⟩」: SURVIVES + domain caveat**。held-out 预测 RMSE 7.8%、LOO 9.6% 完胜 b~c·p 的 86%、partial-F=1934;但仅 **ρ≲0.47 有效**(ρ=0.55 excess 趋平、律失义),且 corr(⟨exc⟩,p)=0.975 → 与 b~0.45p² 近简并,作**参数化经验关系**陈述。

---

## Phase 2d — κ(clip-hardness)第一性原理:Fisher 度规投影（2026-06-16,"继续"轮）

> 脚本 `../d107_paper2_addon/probe3_metric_projection.py`(SLSQP 单纯形投影,real-core K=2 标定 n_eff=412)。

**假设(验证通过)**:约束 KL-MLE = 无约束 MLE 在 **Fisher(G)度规**下投影到可行单纯形 {w≥0,1ᵀw=1},**非欧氏**。agent 的 κ=1 模型用的是一次性欧氏投影(错的度规)。

测 ρ=0.40 meas/sand(σ_proj/σ_unc,无 κ 旋钮):
| K | 欧氏(κ=1) | **Fisher G** | A(widened) | 实测 |
|---|---|---|---|---|
| 5 | 0.951 | 0.928 | 0.945 | 0.89 |
| 6 | 0.883 | 0.819 | 0.869 | 0.88 |
| 7 | 0.852(+23%) | **0.737(+7%)** | 0.829(+20%) | 0.69 |

- **G-度规投影参数自由复现 meas/sand 到 ~7%**(欧氏 +23%@K7)——**clip-hardness 主体 = 度规各向异性沿 worst 模截断更狠**,κ=1 欧氏是 floor。
- **A(widened-data Hessian ∫h_i h_j q^bw/q²)反而更靠欧氏(0.829),证伪"残差来自 KDE 加宽"**——G(信号 Fisher)是对的投影度规。
- **残 ~7%(G 0.737 vs 实测 0.69)= 边界附近超二次 KL 曲率**(真约束 KL-MLE 比其 G-二次近似稍硬);是 named higher-order 效应,非自由旋钮。要闭合须在真 KL(非二次)下投影 = 退回真 NMF(循环),故留作 named 残差。
- ⟹ **κ=1.805 ≈ 欧氏→Fisher 度规修正(可推、参数自由)+ 小超二次残差**。clipping 模型从"1 标定旋钮"升级为"参数自由 Fisher 投影 + ~7% named 残差"。

## Phase 2e — 开放项 #2(p_R / ω_inf / a 干净闭式)结案：不存在（结构已推）

- agent(Phase 2a)已确证:p_R、ω_inf、a 都是 **λ_min(theta-函数次几何下降)的窗口-局部斜率/系数**,随 K-窗与 ρ 漂移 → **干净 ρ-不变闭式本就不存在**(不是没找到)。
- **结构对象已第一性原理推出**:p_R 的 cap = sine-模(Dirichlet 顶模)存活对角分数 W0/Σ|W_k|、W0=c 精确;ω_q² 的 (K−1)/(K+a) = Dirichlet 边缘本征的 Möbius 签名。这些是可写的物理内容;数值常数(0.37/0.92/2.4)是窗口表象,如实标注即可。
- ⟹ #2 **如实结案**:结构推导完成,无干净常数(theta-下降使然)。

---

## ✅ 最终结论（推导完成 2026-06-16,已并入 §0.5a (II) v0.4）

**完整律:excess(K,ρ,bw) = [unconstrained CR(K)·S(K)] × [非负 clipping factor]。**
- **CR(K)=(K−1)^{p/2}**(λ_min 下降幂律),**S(K)=exp(−½bw²ω_q²)**(KDE 对高次模 stretched-exp 压制);积在 K=2..7 像 a+b·ln(K−1) 但**非对数律**(窗口巧合)。
- **clipping**:worst-模单边截断,onset σ_abs_top(K)≈w*_min=0.75/K(K=5@ρ=0.40 穿 O(1)),meas/sand=[g(Λ_K)/g(Λ_2)]^κ;本征值表述 = deg-(K−1)→(K−2) crossover;**K*(n=350)=6,11.1σ**。

### Rigorous / Empirical 账本
| 量 | 状态 | 证据 |
|---|---|---|
| M 精确 Toeplitz,**b=2·erfinv(ρ)²** | **RIGOROUS(精确)** | b_fit/b_theory=1.0000、band-spread ~2e-9 |
| worst 模 = degree-(K−1) contrast | **RIGOROUS(线代事实)** | K=3..7 两 ρ 逐验、overlap 0.978–1.000 |
| sandwich σ_err²=λ_max[G⁻¹ΣG⁻¹] | **DERIVED(~10% vs 真 NMF)** | compute_sandwich |
| S 精确式 + ω_q²=(vAv)/λ_min 控制频率 | **EXACT(exp 式 4–8% Jensen)** | probe_S_* |
| ω_q²=ω_inf²(K−1)/(K+a) Möbius/Dirichlet-edge | **DERIVED(结构)** | R² 0.996–0.9998 |
| **「ln 是窗口巧合」** | **RIGOROUS** | 系数跨窗漂移 34–92%/1328% |
| **p ∝ 1/erfinv(ρ)²** | **仅 RIGOROUS 渐近**(非有限窗律) | top-K 斜率→2;有限窗 ~ρ^{−0.8} |
| clipping onset σ_abs~w*_min=0.75/K + 单边截断机制 | **DERIVED(无参 onset)** | frac/偏度跳变 K=5、11.1σ |
| p_R、ω_inf=0.92Nyq/√κ、a≈2.4/κ²、0.679 deflation(q_S≈0.24 真)、κ=1.805 | **EMPIRICAL**(窗口-局部,因 λ_min theta-下降) | 见 2a / 2c |

**一句话**:深推导把"神秘 ln(K−1) 律"还原为 **Fisher 幂律 × KDE stretched-exp × 非负 clipping** 三机制的乘积;ρ-依赖的严格核 = 符号衰减 **b=2erfinv(ρ)²**;"ln" 与若干前因子是窗口表象/经验标定,已诚实分层。真相源本日志 + main_theorem §0.5a (II)。

---

## 脚本索引（`../d107_paper2_addon/`,共 ~101 个;载重者如下,其余 probe2_* 为 agent 验证脚本）

| 用途 | 脚本 | 产物/作用 |
|---|---|---|
| **框架** | `compute_sandwich.py` | M-估计 sandwich σ_err²=λ_max[G⁻¹ΣG⁻¹],复现实测 excess ~10% |
| **分解** | `compute_factors.py` | excess=CR·S 拆分 + p(ρ)/ω_eff 表(K≤10,多 ρ) |
| **worst-mode shape** | `analytic_worst_mode.py` | 连续 Fisher 最小特征向量 = degree-(K−1)(K=2..7) |
| | `diag_k3_mode.py` / `diag_kge4_mode.py` / `diag_kge4_poly.py` | K=3 / K=4..5 / contrast-多项式投影确认 deg=K−1 |
| **回落/clipping** | `probe_turnover.py` | 经验靶:meas/sand + frac(ŵ<0.01) + w*_min(ρ=0.40) |
| | `probe3_metric_projection.py` ⭐ | **κ=Fisher(G)度规投影**(vs 欧氏/A-widened),κ 第一性原理 |
| **叠图(交付)** | `fig_sigma_err_phase.py` | → `figures/fig_sigma_err_phase_v1.png`;数据 `out_phase/sigma_err_slope_numpy.json` |
| **Phase 1 综合** | `probe_reconcile{,2,3}.py` | λ_min(p∝1/erfinv²)/S(ω_q) 综合裁定 |
| **Phase 2 回落** | `probe2_clip_*.py`(~12) | 单壁截断高斯 + effective-K crossover 两模型 |
| **Phase 2 前因子** | `probe2_{pR,omega,deflation}_*.py`(~12) | p_R cap(sine-模)/ω_q Möbius/q_S 推导 |
| **Phase 2 对抗** | `probe2_{lncoincidence,adv_blaw}_*.py` + `probe2_synth.py` | ln-巧合/b(ρ) 律/p∝1/erfinv² 对抗 + 综合 spot-check |

> 真相源文档:**本日志**(完整推导)、`main_theorem_draft.md` §0.5a (II)(结论)、`worst_mode_fisher_note.md`(shape 成因)。
