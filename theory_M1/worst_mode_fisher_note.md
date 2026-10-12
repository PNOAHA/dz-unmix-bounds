# Worst-mode 解析成因:连续 Fisher 最小特征向量 = degree-(K−1) contrast · note v0.1

> **代号**:M1 §3 / §0.5a (II) 的 worst-mode shape 解析成因。回答"为什么 unpinned 区 worst 切向模是最高次(degree K−1)权重-contrast"。
> **状态**:v0.1(2026-06-15,LN)。**确定性线代**(无噪声/seed),脚本 [`../d107_paper2_addon/analytic_worst_mode.py`](../d107_paper2_addon/analytic_worst_mode.py);K=3 特例与 [`PROGRESS.md`](PROGRESS.md) 旧证(λ_min 本征向量=曲率)自洽 = 独立交叉验证。
> **真相源** [`main_theorem_draft.md`](main_theorem_draft.md) §0.5a (II) v0.4。

---

## 1. 命题

unpinned 区(ρ>ρ_c)实测的 worst 切向模(= ŵ 协方差最大特征向量,直接 NMF 测得 = degree-(K−1) contrast)= **连续 supervised-权重 Fisher 信息在 sum-zero 切空间上的最小特征向量**:

　　**I_ij = ∫ h_i(t) h_j(t) / q_w(t) dt**,　q_w = Σ_k w_k h_k,　h_k = area-normalized 端元密度。

P = I − 11ᵀ/K(投到 sum-zero 切空间),M = P I P 的**最小非零特征向量** = degree-(K−1) 离散 contrast 多项式。

## 2. 数值确认(analytic_worst_mode.py,raw Gaussian σ=25,endpoint w)

| K | ρ=0.47 \|·deg(K−1)\| | ρ=0.35 \|·deg(K−1)\| | dom_deg |
|---|---|---|---|
| 2 | 1.000 | 1.000 | 1 ✓ |
| 3 | 0.998 | 0.999 | 2 ✓(=曲率(1,−2,1)) |
| 4 | 0.997 | 0.999 | 3 ✓ |
| 5 | 0.993 | 0.998 | 4 ✓ |
| 6 | 0.987 | 0.995 | 5 ✓ |
| 7 | 0.978 | 0.992 | 6 ✓ |

最小特征向量与正交化 degree-(K−1) contrast overlap **0.978–1.000**,全 K 全 ρ 主导次数 = K−1。

## 3. 直觉(待严格化)

contrast c(⊥1)对应密度扰动 δq = Σ_k c_k h_k;其 Fisher = ∫ δq²/q ≈ cᵀ I c。**最高次(最振荡、交替号)contrast 让等距重叠高斯的 δq 抵消最多** ⟹ ∫δq²/q 最小 ⟹ Fisher 最小 ⟹ 该方向最难估 = worst 模。这是**信号(raw Gaussian)几何**,与 KDE 带宽 bw **无关**——故 bw 改幅度不改 worst-mode shape(与 bw lens:H 由 σ 构造 bit-identical 一致)。

## 4. 连到 excess(K)

连续 λ_min 随 K 单调降(更多端元→更多抵消空间):
- ρ=0.47:λ_min(K=2..7)=0.607/0.318/0.199/0.142/0.112/0.096 ≈ λ₂/(K−1) ⟹ 纯 Fisher excess √(λ_min(K=2)/λ_min(K)) **≈ √(K−1)**。
- ρ=0.35:λ_min=0.355/0.120/0.0473/0.0214/0.0110/0.00635(比 1/(K−1) **陡得多**)⟹ Fisher 包络更陡;故 excess ρ-依赖(ρ 越小越大)。

实测 excess ~ **ln(K−1)**(对数饱和)比 √(K−1) 慢 ⟹ 差额 = **bw 对高次(更振荡)模噪声的额外抑制**(bw 越大压越狠,bw≈24–30 时 excess<1)。即:**Fisher 几何给"哪个模最弱 + √(K−1) 上包络",bw 噪声抑制把它压成可观测的 ln(K−1)。**

## 5. 仍 open

- ln(K−1) 系数(ρ=0.47 ≈0.51、ρ=0.40 ≈0.76)的第一性原理 + ρ-依赖闭式。
- K≈7 饱和/回落(等距下端元间距 dmu 固定,但 comb 变长;高次 contrast 的 Fisher 随 K 的渐近)。
- bw 噪声抑制对 contrast 次数的依赖(为何把 √ 压成 ln)的解析。

> 这三条是 §0.5a (II) 唯一剩的解析待办;worst-mode **shape** 已闭合(本 note),**magnitude** 半定量(Fisher 包络 × bw 抑制)。
