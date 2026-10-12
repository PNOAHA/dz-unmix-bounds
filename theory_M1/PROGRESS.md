# M1 §3 理论线 · loop 进度记录

> 自定步调 loop 的累积进度。用户 2026-06-11 指示"推进不停",scope 从"只到引理1"扩为完整 §3。
> 每块：草拟 → 对抗自检（6-agent workflow）→ 并入修订 → 落 theory_M1/ → 更新本表。
> 遇 serious-flaw 改不动 / 需用户数学判断的岔路 → 浮出报告。用户说停立即停。

## 状态总览

| 块 | 文件 | 状态 | 对抗自检 | 关键结论 / 遗留 |
|---|---|---|---|---|
| 引理 1（KS→Gram/单纯形条件数）| `lemma1_draft.md` | **v0.3 已增补**（2026-07-26）| ✅ 6-agent 一轮 + v0.3 增补过 3-lens review | 闭式 R=exp(−2erf⁻¹(ρ)²) 机精验证;算法相关指数 **κ_simplex≍ρ^−2(K−2)**（切空间,非完整 R 的 K−1）;K=2 单纯形 κ≡1;**v0.3: (iv)/第二小幂律 K=2..9 确切（charpoly valuation+Newton 多边形）、C_K=2^{K−1}((K−1)!)³/(2K−2)! 精确验 K=2..9** |
| 引理 2（L²–Hessian 谱同阶）| `lemma2_draft.md` | **v0.2 已重构** | ✅ 跑过一轮（6-agent）| 代数全对;**serious-flaw 修复**:"r 与ρ无关"撤销,真正驱动是 1/min_k w_k(边界);只在 **ρ→0病态区 + min_k w_k≥δ** 成立,交付"同阶+指数2(K−2)转移",**不交付可用常数**。**撞岔路:精度目标待主定理需求** |
| (v) 切空间指数解析证明 | `main_theorem §1`+`lemma1 (v)` | **已解析闭合 + γ_K 修正已落地**（2026-07-26 用户批准，lemma1 升 v0.4） | ✅(随主定理round + 3-lens review) | **跨轮矛盾reconcile**:全1是R顶特征向量非底→λ_min(切空间)≍γ_K·λ_min(完整)~ρ^2(K−1);κ_simplex(比值)~ρ^−2(K−2)与λ_min(绝对)~ρ^2(K−1)各司其职、三文件自洽;**✅ 2026-07-26 γ_K 修正已批落地（lemma1 v0.4 + main_theorem §1/§3）:等式偶 K 精确、奇 K ×γ_K=K/(K−1)（K=3 手算 3/2）;lemma1 (v) 机制残句已删,证据存档 lemma1 未决项 8;残留=γ_K 一般奇 K 解析证明** |
| 主定理（σ_err 标度·两模 max）| `main_theorem_draft.md` | **v0.4 已并入**（2026-06-15）| ✅ 6-agent ×2（v0.3 GPU + v0.4 2026-06-15）| **指数 K−2→K−1 修正**(√N·ρ^(K−1)·Δw≳c_K,吃 λ_min 绝对值);K=2"与ρ无关"删(σ_err~1/(√N·ρ));**双渠道** top1≈top1_max(ρ)·[1−P_flip];LAN降格+域 N≫ρ^−2(K−1);病态区=顶点跳跃(待D107);**撞岔路:c_K标定+指数实测+顶点跳跃 须D107受控grid**。**→ v0.4 现状(2026-06-15,均闭)**:σ_err=max(σ_pair,σ_curv) 两模——(I) ρ<ρ_c=C/(√N·ρ₁ₖ)、(II) ρ>ρ_c 曲率主导(slope −1.2…−1.7、C_curv 0.70–0.85、K=3~1.4×K=2)、(III) 非单调峰在 ρ_c≈0.32;1/√N K-无关;c_K=0.48 直接 NMF 标定;(II) K=2..7 实测=**degree-(K−1) worst 模**(K=3 曲率=特例,w/bw-稳)+ excess **对数饱和** a+b·ln(K−1)(ρ=0.47:1.04..1.94,K≈6 平)。过 **2 轮 6-agent 对抗自检**(均抓到真错并修:ρ⁻¹/C_curv 常数/excess 单调/KDE 低通机制)。**worst-mode shape 成因已解析闭合**=连续 Fisher I=∫h_i h_j/q 最小特征向量=degree-(K−1)(K=2..7,worst_mode_fisher_note)。**excess 全律深推导闭合(2026-06-16)=[CR·S]×[非负 clipping]**(b=2erfinv²/「ln 是窗口巧合非定律」/p∝1/erfinv² 仅渐近/clipping onset σ_abs~0.75/K、K*=6@11.1σ);κ 大半第一性原理(2026-06-16)=**Fisher(G)度规投影**(参数自由复现~7%,残~7%超二次 KL);前因子无干净闭式(theta-下降)——**深推导收尾** |
| 推论（解析相图）| `empirical_backbone.md`（实证版 §6）| **实证 v0.2**（2026-06-15）| ✅ 对照 scipy + 受控 + 直接 NMF | 经验简并相图已出（填本槽）;解析版仍 TODO。ρ₅₀√n=1.35(CV4%)验 §0.5a 指数;全球库随机对 0% 稳健简并;S1↔S2 ρ=0.999 对账第三篇。**§6 直接 NMF 标定:C=0.500±0.026(=0.48)、KS边界 ρ√n=1.35≡σ_err≈0.37;K=3 大ρ(>ρ_c≈0.32)floor 上方~1.4×（特征向量证 worst模=curvature u2，结构性；小ρ冻结退回 pairwise）→ 已并入 §0.5a 正文 **v0.4 两模 max**（原式=钉扎区 I，无推翻）。6-agent 对抗自检已过：证伪「全程ρ⁻¹」（大ρ slope−1.2…−1.7、σ_err 非单调峰在 ρ_c）+「C_curv 常数」（→带0.70–0.85）+「ρ_c≈0.2」（→0.32）** |
| **主定理数值验证（σ_err 标度）** | `d107_paper2_addon/sigma_err_scan.py` + `fig_sigma_err_v1` | **本地 K=2/3 实证（2026-06-12）** | ✅ 等价 gate 本地 5/6 PASS | K=2 slope **−1.03**、K=3(N=8e4) **−2.24** 逼近 −(K−1)；发现**三段式 regime**；K=5(−4)+精化+c_K+顶点跳跃待 GPU（`Projects/sigma_err_gpu/` 已就绪）|

## 时间线

- **2026-06-11**：引理 1 v0.1 草拟 → 6-agent 对抗自检 → 挖出 serious-flaw（L² 量冒充算法量）→ v0.2 修订（切空间指数 K−2、删 K=2 伪反常、锁等-σ scope、显式 Lemma 2 依赖）。用户审阅后指示"推进不停"。
- **2026-06-11**：引理 2 v0.1 草拟 → 6-agent 对抗自检 → serious-flaw（"r 与 ρ 无关"假；真正驱动是 1/min_k w_k）→ v0.2 重构（限定 ρ→0 病态区 + min_k w_k≥δ，撤销可用常数声明，改名 Hessian，补注①低权重峰区漏洞，改纯几何支撑集）。**撤销**上轮污染数 κ_Fisher/κ_simplex=2.0/4.0。**发现 grid α=0.3 致 28–53% 样本落单纯形边界**。撞岔路浮出 → **用户拍板:失败指标=top1**（→ α=0.3 红鲱鱼经 margin 消化、引理2 现状够）。
- **2026-06-11**：主定理 v0.1 草拟（top1 口径）→ 6-agent 对抗自检 → 两 serious-flaw（指数 K−2 off-by-one、K=2"与ρ无关"被相图证伪）→ v0.2 修订。**跨轮矛盾 reconcile**：引理1 round 的 κ_simplex(比值)~ρ^−2(K−2) 与主定理 round 的 λ_min(绝对)~ρ^2(K−1) **不矛盾，测的是不同量**；全1是R顶特征向量→切空间删顶非底→λ_min(切空间)=λ_min(完整)~ρ^2(K−1)，主定理吃这个 → **指数 K−1**。(v) 随此解析闭合。**撞 D107 岔路 → 浮出报告：c_K 标定 + 指数 K−1 受控实测 + 顶点跳跃确认 须用户开 D107 grid（相图反解路径已死）。**

- **2026-06-12（数值验证线 / 校级 GPU 准备）**：把主定理 §6 的"D107 受控 grid"待办做成可跑的 σ_err 测量 harness（`sigma_err_scan.py`，三后端 numpy 真实核 / 纯 numpy 参考 / torch GPU）。本地 numpy 实证：
  - **发现三段式 ρ-regime**（理论原只分 ill/LAN 两段）：① ill（N≪ρ^{−2(K−1)}，顶点塌缩、斜率浅）② **LAN（σ_err~ρ^{−(K−1)}，理论窗口）** ③ saturated（ρ 大，触 1/√N 地板，斜率→0）。−(K−1) 只在中间窗口，窗口随 N 增大而展宽（ρ_LAN∝N^{−1/(2(K−1))} 左移，ρ_sat~O(0.1) 固定）。
  - **K=2 干净 −1**（N=2k/20k/80k → −0.92/−0.90/−1.03，全 LAN）。**K=3 完整 crossover**（同三 N → +0.92/−0.76/−2.24，N=8e4 在窗口 [0.09,0.21] 测到 **−2.24≈−2**）。即 §6 待办 **#1 指数 K−1 在 K=2/3 本地实证闭合**。
  - **实验设计两点**（敲死）：(a) 斜率测量必须用 **endpoint w**（k* 在端点）——middle w 的中间端元在 ill 区被几何挤向单纯形顶点、top1 系统塌到 0，污染方差；(b) 那个"中间端元塌缩"恰好是 §6 待办 **#3 顶点跳跃**的现象（`wstar_mean_in_fail` 0.05–0.15），单独用 middle w 演示。
  - **等价 gate**（`validate_sigma_err.py`）：本地 numpy_ref vs 真实核 5/6 PASS，唯一 FAIL 在 K=3 最小 ρ（最 ill cell）cross 微超 2×scatter——**非 bug**（supervised 部分逐字复刻、零差异；差异纯来自数据生成 rng，在 ill 区被放大），判据偏严的统计涨落。
  - **待 GPU（VPN 通了上 L20）**：K=5（−4，需 N~1e6–1e7，本地够不着）+ K=3 精化（N=1e6 加 bootstrap CI）+ c_K 标定（#2）+ 顶点塌缩密采样（#3）。包 + 全量 grid + 容器 validate-first 规矩见 `Projects/sigma_err_gpu/README_RUN.md`。

- **2026-06-12（续·L20 全量 + 判决，⚠ 数值证伪 §3 指数断言）**：校级 L20 web 控制台跑通全量（无 VPN，文件管理上传 + 控制台粘命令）。torch 后端等价已证（同 X 下 supervised 与 numpy_ref **逐位相同**，meanW-diff=0；validate 的 9% 差是 **ρ-恒定的数据生成常数**，K=2 ×0.906/K=3 ×0.93，**不动 log-log 斜率**）。**判决颠覆 §3 核心**：
  - GPU σ_err 实测斜率 **与 K 无关 ~−0.9**（K=2: −0.88 @N=2e3/2e4/2e5；K=3: −0.81 @N=2e5 / **−0.90 @N=1e6**，逐点局部斜率 −0.6~−1.1 震荡，无任何 −2 段）。跨 N 两量级纹丝不动。**不是 ρ^{−(K−1)}**。本地早先 K=3 **−2.24@N=8e4 判明是窗口起点卡 ill→饱和陡降的假象，作废**。
  - 本地线代诊断定位矛盾：**连续 λ_min(R) 与 λ_min(Fisher A=H·diag(1/g)·Hᵀ) 都精确 ~ρ^{2(K−1)}**（K=2/3/5 实测 R:1.99/3.94/7.93、A:1.97/3.81/7.29）⟹ **引理1(iv) 和引理2 的连续版都对**，按 σ_err=1/√(N·λ_min(A)) 该得 ρ^{−(K−1)}。
  - **矛盾 = 连续理论 vs KDE-based 实际方法**。最可能元凶 **固定带宽 KDE(bw=12)**：X 被 bw 平滑（端元 σ_eff=√(σ²+bw²)）、H 是 raw σ；dmu≪bw（小 ρ）时相邻端元被糊合，固定 bw 给 conditioning 装 ρ-天花板，把 ρ^{−(K−1)} 压成 K-无关 ρ^{−1}。**待 bw 扫描验证（假设，未坐实）**。
  - **若坐实 = bounds 核心边界发现候选**（比原 §3 更强）：固定带宽 KDE 把高 K 病态压平成 K-无关；σ_err 由**最弱相邻 pair(ρ^{−1})**主导而非全单纯形 λ_min——呼应第三篇 S1↔S2 pairwise 弱可识别。**§3/引理 正文未动，待用户数学判断改写方向（指数 K−1 → 真实方法 K-无关 ρ^{−1}）。**
  - 数据：`d107_paper2_addon/`(本地)与 L20 `/private/sigma_err/out_{k23,k5,k3big}/sigma_err_slope_torch.json`。

- **2026-06-12（续2·机制坐实 + §3 修订方向已定）**：两诊断闭合因果链，"KDE 假设"由待验证升为坐实：
  - **本征模诊断**（本地 K=3）：σ_err 的 tangent 分解为 **pairwise (1,0,−1)** 与 **curvature (1,−2,1)** 两模。连续 Fisher A 的 λ_min 本征向量 = curvature（overlap 1.000，引理2 连续版对）。dmu>bw 时 curvature std ≪ pairwise（ρ=0.18: 0.0008 vs 0.008）。〔小 ρ 该诊断用 real-core early-stop(tol 1e-7) 跑、curvature 值被欠收敛抬高，不作数；用下条收敛版判。〕
  - **curvature-vs-N 诊断**（L20，固定 ρ=0.05 扫 N=3e4→3e6，收敛版）：σ_err **恒 ~N^{−0.5}**（局部斜率 −0.46/−0.52/−0.49/−0.46），跨 ill→LAN 100× N 标度不变。**curvature 从不 blow up**：N=1e6/3e6 已入 curvature-LAN（N≫ρ⁻⁴=1.6e5），若 curvature 主导 σ_err 该 ~0.4，实测 0.0047（小 85×）；且 σ_err=0.026@N=3e4 = pairwise 外推值。
  - **坐实的完整图景**：连续 λ_min(R)=λ_min(A)~ρ^{2(K−1)} 是对的线代但**不是真实方法有效 Fisher 的 λ_min**。**KDE(bw=12) 把 curvature 模的有效 Fisher 从 ρ⁴ 抬升到 ~ρ²**（bw 以下的二阶结构被平滑、ρ⁴ 小值被托住）⟹ 真实有效 λ_min~ρ²（pairwise，K 无关）⟹ **σ_err 由 pairwise 封顶 ~ρ⁻¹/√N，与 K 无关；ρ⁻⁽ᴷ⁻¹⁾ 灾难两头堵死（需不可达 N≫ρ⁻²⁽ᴷ⁻¹⁾ + 进 LAN 也被 KDE 抬平），永不显现**。⟹ **方法对 K 鲁棒，真瓶颈=相邻 pairwise 简并 = 第三篇 S1↔S2，两篇统一。**
  - **§3 修订方向（定，待起草）**：引理1/2 保留为**理想 density Fisher 的 worst-case 下界**（严格对，给"最坏可能 ρ^{2(K−1)}"）；**主定理改写**为真实 KDE 方法的**可观测** σ_err ~ρ⁻¹（pairwise 封顶、K 鲁棒），显式说明 ρ⁻⁽ᴷ⁻¹⁾ 不可达 + KDE 抬平机制。**实验忠实(dz-unmix-cg = raw-H + KDE bw=12)、σ_err 定义正确；改的是「哪个模主导有限-N 误差」的隐含假设，非实验、非引理代数。** 这把论文从"over-claim 一个观测不到的高 K 灾难"救成"真实方法 K-鲁棒 + pairwise 瓶颈"。

- **2026-06-12（续3·v0.3 落地 + applicability_scan 收敛审查）**：① **v0.3 主定理已落 `main_theorem_draft.md`**（标题改「权重估计误差标度」；§0.5 数值裁定表 + §0.5a 修订主定理「σ_err≍c/(√N·ρ)、K 无关、曲率及以上自由度冻结为 deterministic」+ 机制 + 物理结论 + 强度边界 + §0.5b + 待细化）；v0.2 §1–§6 保留为理想 density-Fisher worst-case 下界、不再是主定理。② **(b) applicability_scan 收敛审查**：主指标 **top1_dom 不受 n_iter=500 欠收敛影响（500=4000=16000 逐位相同：K2sep.6=.915/K3sep.6=.651/K5sep.6=.588/K5sep1=.807）→ 第二篇经验相图安全、不需重跑**；MAE 仅最 ill cell（K=5 sep0.6）+0.012 且**收敛后变大**（= 曲率冻结的独立佐证，非欠收敛虚高）；红线不改 dz-unmix-cg（第一篇真实端元良态、500 步足够）。③ **纯数学待办（LN 可续，无需数据）**：曲率冻结 0 方差的解析证明、top1 失效阈值改 pairwise 口径 √N·ρ·Δw≳c、c 标定。**→ 自包含交接包 `HANDOFF_v0.3_math.md`（给新 session/Fable 接手解析；含 F1–F5 数值约束 + 三待办精确陈述 + 陷阱 + 复现指针）。**

- **2026-06-13（HANDOFF 待办 1 = 曲率冻结解析证明 落地，⚠ 机制更正）**：新 session（Fable）接手 `HANDOFF_v0.3_math.md` 待办 1。**4 数值指纹实验（N1–N4，本地 numpy float64）+ 3 独立推导（KKT / 矩-几何 / rank-1 模代数）各过一轮 adversarial 审 + 2 轮独立 Gauss-Hermite 复核** → 落 **`curvature_freezing_proof.md` v0.1**。判决：
  - **机制 = 边界钉扎（simplex active-set collapse），不是 KDE 低通**。HANDOFF 待办 1 提示的「低通/傅里叶截断」路径被 **K=4 判别实验（N3）证伪**：低通预测极小点内部+曲率冻结；钉扎预测极小点边界+曲率按 v·d 活/冻。N1 中权重塌 10⁻²⁶…10⁻¹³¹（边界），N3 曲率**活**(1/√12)、内部一阶差**冻**（与低通对角相反）。
  - **真实机制**：KDE 确定性 +bw²（每 realization 都加，§3.1）→ raw-σ 字典方差够不到（方差亏损判据 bw²>Δμ²·D，§3.3）→ 总体 KL 极小塌**极端二顶点面** {1,K}（KKT slack η_j>0 认证，§3.4）→ 切空间只剩 **1 活模 d=(e₁−e_K)/√2** → Cov(ŵ) **rank-1**（§3.5）→ ⊥d 模方差冻结（§3.6）。ρ_c=erf(bw/(2√2σ√D)) 闭式，K=3 N1 给 0.29 吻合。
  - **严格度分级**（对抗自检裁定）：【严格】+bw² 恒等式、方差-LP 极端端元唯一最大、rank-1 模代数（机精 1e-16）；【小ρ主阶】均值匹配 a、ρ_c 判据（bw≤12 准 1.3–9.4%、bw=24 过预测 ~25% 因忽略峰度）、对称 K=3 闭式 slack η_mid=Δμ²·deficit/(2σ⁴)（常数 ½ 精确）；【数值认证·解析 open】有限ρ 严格正 slack（min η=1.11e-6>0）、全 KL 泛函边界选择、非对称/K≥4 η 前因子。
  - **⚠ 对 §0.5a/F4/HANDOFF 的修正（curvature_freezing_proof.md §6，待用户确认，未擅改正文）**：(1) F4「曲率方差=0」**K=3 对、K≥4 错**——K=4 曲率活(1/√12)；正确不变量「v 冻结 ⟺ v₁=v_K（⊥ 极端对角）」。(2) 机制名「低通」→「active-set 钉扎」。(3) 活模身份「相邻 pairwise e_i−e_{i+1}」→「极端对角 e₁−e_K」。(4) 冻结指**方差**非均值（曲率均值=常数 1/√6）。
  - **待办 2（top1 pairwise 口径）+ 待办 3（c 标定）**：本证明给「1 活模」直接支持待办 2；幅度 Var(β)~ρ⁻²/N 留待办 3。
  - 脚本 `scratch_curvfreeze/`（n1–n4 + ref_d1/2/3）。

- **2026-06-13（续·待办 2/3 = top1 阈值 + c 标定 落地，⚠ 两处 HANDOFF 措辞更正）→ `pairwise_threshold_calibration.md` v0.1**：承接待办 1（active-set 钉扎）。3 数值标定实验（C1 前因子 vs K / C2 raw-vs-KDE Fisher / C3 top1 分裂，本地 numpy）+ 约化 Fisher 解析闭式 + 确定性核验。
  - **待办 3（c 标定）**：σ_err = **C/(√N·ρ₁ₖ)，C≈0.48 K-无关**（ρ₁ₖ=erf((K−1)Δμ/2√2σ)=极端端元对 KS 距离≈(K−1)ρ）。**C1 决定性**：(K−1)·c_K 平 0.467–0.508（<9%）、c_K∝(K−1)^{−0.92}、σ_err/std(â)=√2 精确（rank-1 沿极端对角）。解析：约化 2-极端端元 **I_red=∫(h₁−h_K)²/q₀≈2π((K−1)ρ)²**（核验 ratio 0.9915）⟹ C_naive=1/√π=0.564。**KDE 修正 κ=σ_meas/σ_naive=0.847±0.031（ρ,K-flat）**⟹ C=κ/√π=0.478=实测。**待办 3 ⚠「KDE 改前因子」已答：是干净 ρ,K-无关 ≈0.85 乘子（−15%），不改标度。**
  - **待办 2（top1 阈值）**：top1 按 k\* 位置**分裂**（C3-Part1 坐实）：**endpoint k\*** survive → â 越 ½ 翻转，阈值 **√N·(K−1)·ρ·Δw ≳ c_top1≈0.34**（c_top1=C/√2，Δw=a_∞−½）；**interior k\*** 被钉扎杀死 ŵ_{k\*}=0 → **top1=0（vertex jump，= §3/§5「顶点跳跃」的真机制）**，ρ>ρ_c 才恢复。top1_confirm 核验：K=3 ρ=0.08 N=300 top1=0.743 vs 预测 0.742（Gaussian flip 精确）。
  - **⚠ 对 HANDOFF/main_theorem 的更正（pairwise_threshold_calibration §6）**：(1) 待办 2「√N·ρ·Δw≳c（K 无关）」→ **√N·(K−1)·ρ·Δw≳c**（带 (K−1)，且仅 endpoint）。(2) 待办 3「连续 A 中间本征值~ρ²」→ **约化极端对 Fisher 2π((K−1)ρ)²**。(3) §0.5a「σ_err K-无关」：**指数**对、**前因子 adjacent-ρ 下含 1/(K−1)**——K-无关只在 ρ₁ₖ 变量下成立；建议正文写 σ_err=C/(√N·ρ₁ₖ)。
  - 脚本 `scratch_pairwise/`（c1_prefactor/c_calib_analytic/top1_confirm）。

- **2026-06-13（续2·用户拍板 → 三待办修正全部并入 main_theorem §0.5a 正文 v0.3.1）**：用户确认 proof §6 + calibration §6 内容后，把待办 1（F4/机制/活模/冻结措辞）+ 待办 2（top1 阈值带 (K−1) + endpoint/interior 分裂）+ 待办 3（c=κ/√π、约化极端对 Fisher）一起并入 `main_theorem_draft.md` §0.5a 正文（升 v0.3.1）。
  - **确认 pass 中新捕获一处 v0.3 自身的错并一并修正**：§0.5 数值裁定表 row 3 把「顶点跳跃」整体判为「欠收敛 transient」**不准**。须分两个量：① 曲率模**方差**在 n_iter 300/1000 的虚高 = 真 transient（收敛后→0）；② **interior k\* 的 top1 顶点跳跃**（ŵ[k\*]→0）= **真 active-set 病态、非 transient**（§0.5b top1_dom 在 500/4000/16000 逐位相同正是其收敛不变佐证）。v0.2 §2 原「顶点跳跃」（k\* 塌 0）**恢复为真现象**——v0.3 错判了它，v0.3.1 复位 + 给机制（钉扎杀内部端元）。
  - **强度边界更新**：#1 K≥4 从「纯机制外推」升级为「K=3/4/5/6 数值坐实」（C1 钉扎+前因子、N2 K3/4/5 KKT、N3 K4 判别）；#3「曲率 0 方差解析证明未完成」→ **已完成**（curvature_freezing_proof）。
  - top1_confirm（K=3 全 + K=4）：top1 实测 vs Φ(z) 逐 cell 吻合 ~1–2%，z 含 ρ₁ₖ ⟹ (K−1) 阈值经 K=3+4 验证。两 proof §6 标记改「已并入正文 v0.3.1」。**三个纯数学待办闭环。**

- **2026-06-14（实证骨架 = §3 推论实证版 + 桥到第三篇）→ `empirical_backbone.md` v0.1**：起因校级 L20 闲置评估 → 落到全球库 all-pairs 相异度这个 O(N²) GPU 瓶颈 → 顺势做成 §3 缺的实证证据（引擎在 `../../dz-global-db/` 三个 `_proto_*.py`，均对照 scipy 验过）。
  - **标度律（验 §0.5a 指数）**：受控两高斯端元（ρ 解析已知），经验 50% KS 不可分边界 **ρ₅₀·√n = 1.35（CV 4%，n=20→300）** → 实证 ρ∝n^(−1/2)，与 σ_err∝ρ⁻¹/√N 同族（ρ√n=const）。是 pairwise(2 源)律，正打在「pairwise 封顶」绑定模上。⚠ 1.35 是 KS 可分性常数、**非** σ_err 前因子 C≈0.48（同几何不同常数）。
  - **简并相图（填「推论」槽）**：成品图面板 A 经验等值线（P=0.5/0.9）∥ M1 iso-σ_err 线 → K×ρ 失败面**实证版**出炉；解析版仍 TODO。
  - **真实全球库定位**：N=300 池 44,850 对，中位 ρ̂=0.72，**随机全球对稳健简并 0.00%、模糊 3.08%**（筛 margin<0.1 尾只 bootstrap 5.7%）。简并是（小ρ+小n）角的条件现象 → 全球物源空间绝大多数良态，与「高 K 灾难观测不到」（§0.5a 物理结论）一致。
  - **S1↔S2 对账第三篇贝叶斯**：ρ_S1S2=0.999（完全可分）。不矛盾、且**收紧**第三篇措辞——S1↔S2 是最近绑定对（相对最软，= 物理结论的相邻 pairwise），但绝对安全 ⟹ 贝叶斯后验展宽**主要是有限颗粒数(n=31–111)、不是端元简并**。建议正文 "owing to end-member degeneracy" → "relatively softest (binding) direction, spread dominated by finite grain counts (ρ_S1S2≈1.0)"。
  - **下一步**：跑 `../d107_paper2_addon/sigma_err_scan.py`（本地 numpy K=2/3）把**实测 NMF σ_err** 叠同一 (ρ,n) 平面，把 iso-σ_err 从理论断言变实测，缝死 bounds §3 与第三篇。算力本地秒级，L20 暂不必要。
  - memory：[[dz-degeneracy-empirical-backbone]]。

- **2026-06-15（直接 NMF 叠图 → §6 实证 → 机制深挖 → 6-agent 对抗自检 → §0.5a 升 v0.4 两模 max）**：接 empirical_backbone §5「下一步」。
  - **直接 NMF σ_err 叠 (ρ,n) 平面**（`d107_paper2_addon/fig_sigma_err_phase.py`，128 cell real 核 R600）：K=2 pairwise 律直接标定 **C=0.500±0.026(=0.48)**、log-log 斜率 +0.95；**一句话缝合 KS 边界 ρ√n=1.35 ≡ σ_err≈0.37**。iso-σ_err 从理论断言变实测，缝 bounds §3 ↔ 第三篇。→ empirical_backbone §6。
  - **机制深挖**（`diag_k3_mode.py` 特征向量）：K=3 大 ρ floor 上方 ~1.4× 的 worst 模 = **curvature u2**（|·u2|≈1.0）。**自我证伪两次**（初判「finite-n 曲率未冻结、大 N 消失」→ 特征向量证为**结构性 curvature 主导、小 ρ 才冻结**）。
  - **6-agent 对抗自检**（5 lens + 综合）：主体幸存（curvature=大ρ worst、结构性 ~1.4×、小ρ 冻结、1/√N K-无关、§0.5a 头条仅覆盖钉扎区）；**证伪 part(c)「全程 ρ⁻¹」**（大ρ ρ-slope −1.2…−1.7、σ_err 非单调峰在 ρ_c）；C_curv→带 0.70–0.85；ρ_c≈0.2→0.32。综合裁定还**纠正了证伪 lens 自己的「ρ-flat」误判**（跨 ρ_c 拟合 regime-mixing 假象）。
  - **用户拍板 → §0.5a 升 v0.4**：单式 σ_err=C/(√N·ρ₁ₖ) 扩成**两模 max**（原式=钉扎区 I，**无推翻**；新增 unpinned 曲率区 II + 非单调过渡 III）。「K 鲁棒」精确化：无 ρ^{−(K−1)} 灾难（铁）、K-依赖两区反号（I 随 K 降 / II 随 K 升）。真实良态端元（Δμ≫Δμ_c≈21 Ma）落 II 区。草案 `sec0_5a_revision_draft.md`（已并入）。
  - **续·K≥4 扩展（第 2 轮 6-agent 对抗自检）**：worst 模 = **degree-(K−1) contrast 多项式**（宇称 (−1)^{K−1}；K=3 曲率 (1,−2,1)=degree-2 特例；K=2..5 overlap 0.98–1.00；w/bw-稳）；excess 随 K **对数饱和** a+b·ln(K−1)（ρ=0.47 K=2..7:1.04..1.94，K≈6 趋平、ρ=0.40 K=7 回落）。**又两处自我修正**：「excess 单调」→ 对数饱和；「KDE 低通机制」→ 证伪、符号相反（bw 大→excess 小，H bw-无关 bit-identical，bw 只平滑数据噪声 σ_err~bw^{−0.5}）。已并入 §0.5a (II)。脚本 `diag_kge4_{mode,poly}.py` + workflow `probe_*`。
  - **续·worst-mode shape 解析成因闭合**（确定性线代，`analytic_worst_mode.py` → [`worst_mode_fisher_note.md`](worst_mode_fisher_note.md)）：worst 模 = **连续切空间 Fisher I_ij=∫h_i h_j/q 的最小特征向量** = degree-(K−1) contrast（K=2..7、ρ=0.47/0.35 overlap 0.978–1.000；K=3 即曲率、与本表旧 λ_min 证自洽=独立交叉验证）。直觉：最振荡 contrast 的 Σc_i h_i 抵消最多→Fisher 最小→最难估；信号几何、bw 无关。连续 λ_min 随 K 降 → 纯 Fisher excess ~√(K−1) 上包络，bw 噪声抑制压成实测 ln(K−1)。**剩 open**：ln 系数 + K≈7 回落解析。
  - 脚本：`d107_paper2_addon/{fig_sigma_err_phase,diag_k3_mode,diag_seed_stability,conv_probe,conv_probe2,diag_w_sweep,judge_rho_exponent}.py`。memory：[[m1-identifiability-theorem-line]]。

- **2026-06-16（excess(K,ρ,bw) 深推导,2 phase workflow + 对抗自检,「不惜代价」）**：闭合 §0.5a (II) 的 ln 系数 / ρ-依赖 / K≈7 回落 open。全推导 [`excess_derivation_log.md`](excess_derivation_log.md)。
  - **框架**：M-估计 sandwich σ_err²=λ_max[G⁻¹ΣG⁻¹] 复现实测 excess ~10%；分解 **excess = CR·S**（CR=(K−1)^{p/2} Fisher 幂律、S=exp(−½bw²ω_q²) KDE 对高次模 stretched-exp 压制；ω_q²=(vᵀAv)/λ_min q-加权频率,解 omega_eff 谜）。
  - **rigorous 核**：overlap M 精确 Toeplitz、衰减常数 **b=2·erfinv(ρ)²**（b_fit/b_theory=1.0000）；worst 模=degree-(K−1)；**「a+b·ln(K−1)」证伪为窗口巧合**（跨窗系数漂移 34–92%、a 变负；CR 幂律 / S stretched-exp β≈0.95–1.16）；p∝1/erfinv² **仅渐近**（有限窗 K=2..7 压扁 ~ρ^{−0.8}）。
  - **回落 = 非负 clipping(ŵ≥0)**：worst-模坐标单边截断,**onset σ_abs_top≈w*_min=0.75/K**（K=5@ρ=0.40 穿 O(1)=frac/偏度跳变位置）；deg-(K−1)→(K−2) crossover；**K*(n=350)=6,11.1σ**；深度 meas/sand=[g(Λ_K)/g(Λ_2)]^κ,κ=1.805 单点标定。
  - **诚实分层**：empirical=κ=1.805(MUR 边界硬度)/p_R(cap=sine-模存活对角、ρ-依赖非普适3/8)/ω_inf=0.92Nyq√κ/a≈2.4/κ²/0.679(全局回归系数、q_S≈0.24 真)。脚本 `compute_{sandwich,factors}.py`+`probe_turnover.py`+`probe2_*.py`(~40)。已并入 §0.5a (II)。

- **2026-07-26（未决项 3 部分结清：(iv)(v) 幂律从数值断言 → K=2..9 确切代数事实 + C_K 闭式；3-lens review 18 findings 落地 + γ_K 新发现）→ lemma1_draft 升 v0.3**〔起源：D107-Mathematica 用途对抗审查的副产品——审查 agent 为证伪"仅 Mathematica 能算"当场用 sympy 完成核心计算；本 session 两路独立复核后写入，再过 3-lens 对抗 review〕：
  - **确切谱阶梯**：R(1−u) 的 charpoly 在 ℤ[u] 精确展开，λ^{K−j} 系数 valuation 恒 = j(j−1)/2（K=2..9）⟹ Newton 多边形斜率 0..K−1 各一 ⟹ λ_{(k)} ≍ C_{K,k}·u^{k−1} 对这些 K 确切。(iv) 的 K−1 与第二小 λ_{(K−1)} 的 K−2 均落此。**一般 K 仍开放**（正文若按一般 K 陈述则该开口仍必修，见 lemma1 未决项 3）。
  - **前导常数（精确有理，K=2..9）**：λ_min：1, 4/3, 12/5, 192/35, 320/21, 3840/77, 26880/143, 114688/143（v0.2 的 1/1.33/2.4/5.49 即前四项浮点影子）；第二小：2, 4, 8, 96/5, 384/7, 1280/7, 7680/11, 430080/143。**闭式识别：C_K = 2^{K−1}((K−1)!)³/(2K−2)!**（相邻比 K²/(2K−1)；八点精确 match；一般 K 待证）。
  - **结构**：R=D·W·D 广义 Vandermonde（t=q^{−2}），det R = q^{2Σi²}·t^{K(K+1)/2}·∏_{i<j}(t^j−t^i)（K=3..5 符号恒等验证）；det 幂次 K(K−1)/2=Σ(k−1) 与阶梯相容（v0.1"det 错配"疑虑彻底解除）；K=5 常数核：∏_k C_{5,k}=294912=sf(4)·2¹⁰ ✓。
  - **两路验证**：① sympy 1.14 精确有理（valuation K=2..8 共 1.9s / 主子式阶梯 K=2..7 共 10s / K=8,9 charpoly-e_k 路 1.2/2.3s）② mpmath 自适应精度数值特征值，**抽查 K=5,7** 于 u=1e-6/1e-8：K=5 worst 6.3e-6→6.3e-8、K=7 1.35e-5→1.35e-7，**均严格 ∝u**。⭐教训：初版 flat dps=50 在 K=7,u=1e-8 触精度地板（假偏差 1.16e-5 不随 u 降），review 抓出后改自适应 dps 重跑——**高精度对拍前先估 λ_min 量级 vs dps 噪声底**。脚本入库 `theory_M1/verify/`（4 个 .py）。
  - **文献锚（仅条目存在性+标题/摘要核验，正文未读，2026-07-26）**：Barthelmé & Usevich 2021, *Spectral properties of kernel matrices in the flat limit*, SIAM J. Matrix Anal. Appl. 42:17–57, DOI 10.1137/19M129677X——据摘要覆盖解析径向核平坦极限特征值主阶 + 离散正交多项式特征向量；一般 K 指数**有可能**查引可得，**是否覆盖等间距高斯核前导常数尚未核对**。进正文引用前须下载原文核对定理编号与常数约定。
  - **⭐ γ_K 新发现（review 实测 + K=3 手算复核，未改任何原文）**：切空间（Helmert Z）实测（dps=150，K=2..8）裁决了机制句之争——①切空间底指数 = K−1（追踪 λ_min(R)），机制句"底追踪第二小（阶 K−2）"**证伪**；②切空间顶 = 次大 λ_{(2)}(R) **精确**（反对称宇称 1ᵀv=0）；③但 reconcile 等式 λ_min(ZᵀRZ)=λ_min(R) 只在**偶 K 精确**（宇称），**奇 K 常数×γ_K=K/(K−1)**（1.5/1.25/1.1667 实测；K=3 手算=3/2 解析复核）⟹ λ_min(ZᵀRZ) ≍ γ_K·C_K·u^{K−1}。κ_simplex 指数不变；**主定理吃切空间 λ_min 绝对值处奇 K 需补 γ_K**。全部证据在 lemma1 未决项 8。**（续，同日）用户批准建议改法 → 已落地，lemma1 升 v0.4**：机制残句删除并重写 γ_K 版、证明 (v) reconcile "="→"≍ γ_K"（补 Cauchy 交错 + centrosymmetric 宇称两步论证）、main_theorem §1（状态行/表格/机制/结论）+ §3(2) 同步补 γ_K、λ₂ 裸记号触及处统一 λ_{(k)}；残留=γ_K 一般奇 K 解析证明 + 旧文记号未系统扫。
  - **顺带覆盖**：main_theorem §6"纯数学"待办「§1 reconcile 严格化（λ₂~ρ² 证明）」被阶梯覆盖（次大 λ_{(2)}≍C_{K,2}·u、u≍(π/2)ρ²，K=2..9 确切）——**✅ 已批关闭（2026-07-26，main_theorem §6 已划；此前笔误 §5 已更正）**。

- **2026-07-26（CAD 探针·获批半天实验：K=2 理想下界 moment 代理边界 → 精确公式）→ `scratch_cad_probe/`（脚本+输出+sympy 复核）**：D107 .200 单 kernel Mathematica 15.0；CylindricalDecomposition/Reduce 全部**秒级**返回（远低于 1800s 门槛），**双杀活判据通过，探针存活**。三件精确结果（均过 sympy 第二路 + mpmath 数值第三路复核）：
  1. **均值统计量失效边界精确化**：failure ⟺ τ < √(s(1+(1−a²)d²))/(2d)（a=1−2w、s=z²/N；CAD 可读区域 0.7s + 单调性 True 证书）；ρ 坐标 τ*(ρ)=√(s+2(1−a²)s·erfinv²ρ)/(2√2·erfinvρ)。**小 ρ 展开 ρ·τ*·√N/z = 1/√(2π) + (√(π/2)/6)ρ² + O(ρ³)** ⟹ σ_err 口径（×√2）= **1/√π：C_naive=0.564 从数值拟合升为精确常数**，并附精确次阶系数；I_mean(ρ)=2πρ²−(2π²/3)ρ⁴+O(ρ⁶)，首项精确 = I_red。量级 gate：ρ=0.1 处 0.401，与真实方法 c_top1≈0.34 差 ~15%（与 KDE κ≈0.85 修正同向同量级）。
  2. **单高阶矩一致劣于均值（k=3,5,7 精确证明，任意 w）**：比较式 a² 项恒等消去 ⟹ 与 w 无关；差 = −2d²(4d⁴+15d²+3)（k=3）等显式负定多项式，CAD 区域 = False。审稿弹药："为何一阶统计量已足够"有精确答案。
  3. **双矩 (m₁,m₃) 精确 Fisher 增益**：gain = 2d⁴/(3+3(4−a²)d²+9(1−a²)d⁴+2(1−a²)d⁶)，处处 >0（Reduce 证书）但小 d 仅 (2/3)d⁴+O(d⁶)（d=1 恰 1/13≈7.7%、病态区 d=0.25 仅 0.2%）⟹ **病态区均值单矩渐近携带全部 Fisher 信息**，理想 σ_err~1/(√N·ρ) 律由一阶矩单独实现。
  - **诚实定位（按获批 scope）**：全部结果属**理想下界层**，不改真实方法结论（KDE/钉扎/clipping 主导，2026-06-12 GPU 已定）；落点 = 本条 + CAGEO-D-26-01238 审稿弹药，**非新论文线**。CAD 使用感照实记：本问题多项式结构简单，Factor/Reduce 即可替代，"CAD 独有价值"未真正兑现。可选追加（≤2 天 surrogate 阶数扫描）不自动启动。
  - ⚠ **教训（这次撒谎的是 sympy）**：sympy 对 erfinv 复合表达式的 series 次阶系数给错（√(2π)/8 vs 正确 √(π/2)/6），mpmath 40 位数值仲裁判 Mathematica 对——**series 级对拍也要数值第三路仲裁**。

## 给用户的待审/岔路清单（累积）

1. ~~(v) 指数 K−2 现为数值断言 — 要不要补解析证明（已纳入 loop 队列）~~ **◐ 部分闭（2026-07-26）**：K=2..9 确切代数事实（精确 charpoly + Newton 多边形）+ C_K 闭式（lemma1_draft v0.3 增补段，两路验证 + 3-lens review）；残留=一般 K 统一证明（**正文若按一般 K 陈述则仍必修**；文献锚 Barthelmé & Usevich 2021 仅摘要级核验）。~~新岔路（未决项 8）~~ **✅ 已批落地（2026-07-26，lemma1 v0.4 + main_theorem §1/§3）**：机制残句删除重写 γ_K 版、reconcile "="→"≍ γ_K"（Cauchy 交错 + 宇称两步论证）、主定理引用处补 γ_K；残留=γ_K 一般奇 K 解析证明（现 K=3,5,7 实测 + K=3 解析）。
2. **引理 2 常数依赖工作点 w** — Fisher 依赖 w、R 不依赖;近单纯形边界常数退化。可接受性待用户判。
3. **等-σ 硬假设** — 真实 DZ 端元 σ 不等,是适用性边界还是要推广（future work vs 正文）。
4. ~~待办 1/2/3 机制更正待确认~~ **✅ 已闭（2026-06-13）**：用户拍板后三待办修正（机制=active-set 钉扎、F4 K=3 巧合、σ_err 前因子含 1/(K−1)、top1 阈值带 (K−1)+endpoint/interior 分裂、顶点跳跃复位为真现象）全部并入 main_theorem §0.5a 正文 v0.3.1。两 proof 文件 §6 标记「已并入」。**M1 §3 三个纯数学待办全部闭环。**
5. **(v0.4,部分闭 2026-06-15)** unpinned 区 open 进展:(a) ~~K≥4 未测~~ **已测 K=2..7**(degree-(K−1) worst 模 + excess 对数饱和 a+b·ln(K−1));(b) worst-mode **shape 成因已解析闭合** = 连续切空间 Fisher I=∫h_i h_j/q 的最小特征向量 = degree-(K−1)([`worst_mode_fisher_note.md`](worst_mode_fisher_note.md),K=2..7 overlap 0.98–1.00,确定性线代)。(c) **excess(K,ρ,bw) 全律已深推导闭合(2026-06-16,2 phase workflow + 对抗自检)** = [CR·S]×[非负 clipping]:M Toeplitz **b=2erfinv²**、**「ln 是窗口巧合非定律」**、clipping onset σ_abs~w*_min=0.75/K、K*=6@11.1σ;p∝1/erfinv² **仅渐近**。并入 §0.5a (II)+[`excess_derivation_log.md`](excess_derivation_log.md)。**(2026-06-16「继续」轮收尾)**:κ **大半第一性原理** = 约束 MLE 在 **Fisher(G)度规投影**(非欧氏),参数自由复现 meas/sand 到 ~7%(欧氏 +23%@K7),残 ~7%=边界超二次 KL 曲率(named);p_R/ω_inf,a 的干净 ρ-不变闭式**确证不存在**(theta-下降,结构对象 sine-模/Dirichlet-edge 已推)。**深推导线实质收尾。**
