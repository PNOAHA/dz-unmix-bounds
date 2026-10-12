# v0.3 数学接手包 · 给新 session（Fable）

> **给接手者**：你是新开 session 的数学助手，没有上一轮对话记忆。本文件自包含。
> **背景轨迹真相源**：[`PROGRESS.md`](PROGRESS.md)（2026-06-12 四条续记，完整数值-验证过程）；主定理现状：[`main_theorem_draft.md`](main_theorem_draft.md) v0.3 §0.5/§0.5a。
> **先读**：本文件 → `main_theorem_draft.md` §0.5（10 分钟），即可进入。
> **你的任务**：把 v0.3 的三个**纯数学**待办做掉（不需要 GPU/数据，数值事实已全部坐实，列在 §2）。

---

## 1. 一页背景

`dz-unmix-bounds` 第二篇方法论文 §3，代号 M1。原计划（v0.2）：证明 supervised KL-NMF 解混的权重估计误差 σ_err ~ ρ^{−(K−1)}（端元数 K、相邻端元 KS 距离 ρ），即"端元越多越病态"。

**2026-06-12 GPU 全量数值验证把 v0.2 的指数 K−1 证伪了**，并坐实了正确图景（v0.3）：

> 真实 KDE-supervised 方法（先把 N 颗锆石年龄做 KDE 成密度 X，再用 raw-σ 高斯端元字典 H 做 KL-散度乘性更新求权重 w）的可观测 σ_err **与 K 无关 ~ ρ⁻¹/√N（相邻 pairwise 模封顶）**；二阶及以上差分模（曲率）的统计方差**冻结为 0**。理想 density-Fisher 的 ρ^{−(K−1)} 是 worst-case 下界，对真实方法**不可达**。

**为什么是更好的论文**：从"over-claim 一个观测不到的高 K 灾难"变成"真实方法对 K 鲁棒、瓶颈是相邻 pairwise 简并"——与第三篇贝叶斯实测的 S1↔S2 弱可识别统一。

---

## 2. 已坐实的数值事实（你要解析地解释/证明这些）

设 K 个**等-σ 高斯密度** h_k(t)=N(t;μ_k,σ)，峰位**等间距 Δμ**，相邻 KS 距离 ρ = erf(Δμ/(2√2σ))。混合 g(t)=Σ_k w_k h_k(t)。

| # | 事实 | 已验证方式 |
|---|---|---|
| F1 | 归一化 L² Gram R（R_ij=⟨h_i,h_j⟩/‖h_i‖‖h_j‖）的 **λ_min ~ ρ^{2(K−1)}** | 数值 K=2/3/5 斜率 1.99/3.94/7.93；λ_min 本征向量 = **曲率二阶差模** (1,−2,1,…) |
| F2 | 连续 Fisher **A = H·diag(1/g)·Hᵀ** 的切空间 λ_min **也 ~ ρ^{2(K−1)}** | 数值 K=2/3/5 = 1.97/3.81/7.29（support-masked 积分）。⟹ 引理1/2 连续版**严格对** |
| F3 | 真实 KDE-supervised（充分收敛）的 **σ_err ~ ρ⁻¹/√N，与 K 无关** | GPU：K=2 斜率 −0.88、K=3 −0.90，跨 N=2e4→1e6 不动（K=3 若真 ρ⁻² 应 −2） |
| F4 | **曲率模 (1,−2,1) 的统计方差收敛到精确 0.00000** | 固定数据 X 扫 n_iter：曲率 std 在 n_iter≥4000 = 0.00000（300/1000 步的大值是欠收敛 transient）；pairwise (1,0,−1) std 稳定非零 |
| F5 | pairwise 模 std ~ **1/√N**（且 ~ρ⁻¹） | N=5e4→2e5 给 σ_err 0.0197→0.0093，比值 0.47≈1/√4 |

**核心张力（你要解析地解决）**：F1+F2 说连续理论的最坏方向是曲率模、λ_min~ρ^{2(K−1)} ⟹ 若 σ_err=1/√(N·λ_min(有效Fisher)) 该得 ρ^{−(K−1)}。但 F3+F4 说真实方法 σ_err~ρ⁻¹（pairwise）、曲率模 0 方差。**结论：真实 KDE 方法的有效 Fisher 的 λ_min 不是连续 A 的 ρ^{2(K−1)}，曲率方向被 KDE 抹成 deterministic。**

---

## 3. 三个数学待办

> **✅ 待办 1 已做（2026-06-13）→ [`curvature_freezing_proof.md`](curvature_freezing_proof.md) v0.1。** ⚠ 机制更正：不是下面"可能路径"提的 KDE 低通，而是 **active-set 边界钉扎**（K=4 判别实验 N3 证伪低通）。且本待办下面"二阶及以上差分趋 0"的陈述**对 K≥4 错**——K=4 曲率模是活的（=pairwise 的 1/√12），内部一阶差才冻结；正确不变量「v 冻结 ⟺ v₁=v_K（⊥ 极端对角 e₁−e_K）」。详见 proof §6（待用户确认并入正文）。下文保留原始陈述供对照。

### 待办 1（核心，最有意思）：证明曲率冻结 = 0 统计方差

**精确问题**：观测 X = ĝ_bw（g 的 KDE，带宽 bw，N 个 i.i.d. 样本 ~ g，**bw ≳ Δμ**）。估计 ŵ = argmin_{w∈Δ^{K−1}} D_KL(X ‖ Σ_k w_k h_k)，字典 = **raw-σ** 端元 h_k（注意 mismatch：数据的有效端元是 h̃_k = h_k ⊛ K_bw，宽度 σ_eff=√(σ²+bw²)，但拟合用 raw h_k）。

**要证**：在 sum-to-one 的 (K−1) 维切空间中，ŵ 绕其期望的协方差，
- 沿**一阶差分**方向 (e_i − e_{i+1}) 非零，标度 ~ 1/(N·ρ²)（⟹ σ_err~ρ⁻¹/√N）；
- 沿**二阶及以上差分**方向 (e_i − 2e_{i+1} + e_{i+2}, …) **趋 0**（与数据 X 涨落无关）。

**可能路径**（供参考，不限）：
- 线性化收敛点的 KKT：δŵ = M·δX，M = (HᵀW_g H)⁻¹HᵀW_g（W_g=diag(1/g) 的某离散版），δX = KDE 涨落。Cov(ŵ)=M·Cov(X)·Mᵀ。**关键**：KDE 涨落 Cov(X) 是低通（截断频率 ~1/bw）；曲率模对应频率 ~1/Δμ > 1/bw（因 bw≳Δμ）→ 落在 KDE 截止外 → δX 在曲率方向无投影 → Cov(ŵ) 曲率方向 = 0。需把这个"低通截断 → 曲率方向 0 方差"做严格（傅里叶域 + raw-H vs h̃ mismatch 的处理）。
- 或：直接证 raw-H 拟合 KDE-数据 的灵敏度矩阵 M 在曲率方向与 Cov(X) 正交。
- 注意"等-σ 高斯 + 等间距"是可用的强结构（h_k 是平移族，Gram 是 q^{(i−j)²} 形式，q=e^{−Δμ²/4σ²}）。

> **✅ 待办 2+3 已做（2026-06-13）→ [`pairwise_threshold_calibration.md`](pairwise_threshold_calibration.md) v0.1。** ⚠ 两处下面的陈述要更正：(2) 阈值**不是「√N·ρ·Δw≳c K 无关」**，而是 **√N·(K−1)·ρ·Δw≳c**（带 (K−1)，来自极端端元对 ρ₁ₖ≈(K−1)ρ，非相邻），且**仅 endpoint k\***；interior k\* 是 vertex jump（top1=0）。(3) c 标定对象**不是连续 A 的中间本征值**，而是**约化 2-极端端元 Fisher I_red=2π((K−1)ρ)²**；c=κ/√π，κ=0.85（KDE 前因子修正，ρ,K-flat）。下文保留原始陈述供对照。

### 待办 2：top1 失效阈值改 pairwise 口径

v0.2 §2 的主定理是 top1（argmax 正确率）阈值 **√N·ρ^{K−1}·Δw ≳ c_K**（Δw=argmax margin）。曲率口径错。argmax 翻转沿**相邻端元混淆方向 = pairwise**，故改为
**　√N·ρ·Δw ≳ c（K 无关）**。
需从"pairwise 模 σ_err~ρ⁻¹/√N + argmax 翻转需误差越 Δw/2"重推 top1 阈值（v0.2 §3 的双渠道分解 top1≈top1_max(ρ)·[1−P_flip] 框架可复用，把 P_flip 的标度从 ρ^{K−1} 改 ρ）。

### 待办 3：c 标定

σ_err(pairwise) = c/(√N·ρ)。c = pairwise 一阶差模的 Fisher 前因子。从连续 Fisher A 的**中间本征值**（非 λ_min；K=3 时 A 的本征值为 {O(1), ~ρ², ~ρ⁴}，pairwise 对应 ~ρ² 那个）解析算前因子，或数值标定。⚠ KDE 可能改前因子（不只标度）——若解析 c 与 GPU 实测前因子不符，差异即 KDE 对 pairwise 模的修正，本身是结果。

---

## 4. 已知陷阱（别重蹈，都踩过了）

1. **F1/F2 是对的，别再质疑** λ_min(R)、λ_min(A) ~ρ^{2(K−1)}。错的不是这个，是"它驱动 σ_err"。它是理想下界。
2. **收敛性是命门**：σ_err 必须用**充分收敛**的 supervised（real-core init=seed-0 uniform 在 ill 区需 n_iter≥4000；GPU torch init=1/K 因曲率投影恒 0 巧合 500 步即收敛）。n_iter=300/1000 的"曲率大方差"是**欠收敛 transient 假象**，不是真病态（这点害我绕了一圈）。
3. **σ_err 是统计 precision（绕样本均值的 std），不是 accuracy**。曲率方向有 deterministic **bias**（raw-H vs h̃ mismatch），但 0 **variance**。两者别混。
4. **强度边界（须写进论文）**：数值只验了 K=2/3（都 ~−0.9）；K=5 数据非单调不可用（N 远不及曲率-LAN ρ⁻⁸~1e8）。"K 无关 / K≥4"靠机制外推。斜率是 ρ^{−0.9±0.1}（R=500 噪声），非"恰好 ρ⁻¹"。
5. **红线**：不改 `../dz-unmix-cg/`（第一篇 DOI reserved 10.5281/zenodo.20371892）。其 `supervised_unmix` 默认 n_iter=500 在真实端元（间距大、良态）足够；已查 bounds applicability_scan 的 top1_dom 主指标不受欠收敛影响（§0.5b）。

---

## 5. 文件指针 + 数值复现

- `main_theorem_draft.md` v0.3 — §0.5 数值裁定 / §0.5a 修订主定理 / §0.5b 收敛审查；**v0.2 §1–§6 保留为理想 density 推导**（你可引用其引理1/2 代数）。
- `lemma1_draft.md` — KS ρ → Gram λ_min~ρ^{2(K−1)}（80 位验证过）。
- `lemma2_draft.md` — L²→Hessian 谱同阶（连续版对，KDE 修正是新东西）。
- `PROGRESS.md` — 2026-06-12 四条续记 = 完整数值轨迹（含每个诊断的命令与数字）。
- **数值复现脚本**（本地 numpy，`C:/Users/Administrator/miniconda3/python.exe`，需 sklearn）：
  - `../d107_paper2_addon/sigma_err_scan.py` — σ_err harness（三后端 numpy/numpy_ref/torch）。
  - F1/F2 的 λ_min 标度、F4 的 n_iter 扫描、本征模分解：命令都在 PROGRESS.md 2026-06-12 续记里（可直接粘）。
- GPU 全量数据：L20 容器 `/private/sigma_err/out_{k23,k3big,curvN}/sigma_err_slope_torch.json`（需 VPN/web 取回）；GPU 包 `../../sigma_err_gpu/`。

---

*留痕 by LN，2026-06-12。数值部分已闭环坐实；交给你做解析。待办 1 是主菜。*
