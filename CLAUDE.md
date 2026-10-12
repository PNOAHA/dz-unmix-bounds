# DZ-Unmix Bounds · 项目级 Claude 指引

> 自动加载。**用户**（潘路加）不用看；这份是给 **AI 助手** 的项目背景。
>
> ⚠⚠ **2026-08-18 状态 banner（压倒下文旧说法）**：本地这个目录 = **过期 v0 stub 骨架 + E 线寄生素材**，论文实验与成稿早已不在这里发生。
> - **B 论文现状**：C&G（CAGEO-D-26-01238）07-31 desk reject → **2026-08-05 已投 MDPI Minerals（minerals-4514253）**，08-07 疑遭桌面拒待核邮箱。稿件权威目录 = `../2024河北省教育厅项目/第二篇_方法适用边界/`（续工先读其 00_RESUME 顶部 2026-08 块）。
> - **真代码仓** = GitHub PNOAHA/dz-unmix-bounds **v1.1** + Zenodo concept 10.5281/zenodo.20768131（当前版本 21739169）；工作 clone 在 OneDrive 外（RESUME 记的 Administrator 路径，换机后用前先 Test-Path）。**本目录勿当投稿代码。**
> - **E 线寄生在此**：`d107_paper2_addon/` = E 线（Bayesian UQ，≠第二篇；已投 C&G → 07-15 Sent Back 未拆）素材 + RESUME。做 E 线先读 `d107_paper2_addon/00_RESUME_Bayesian.md`，别被本文件「这是第二篇代码仓」的定位误导。
> - 跨篇真相源 = `../00_论文总览_MASTER.md`（§1 表 + §2-B/§2-E 档案节已 2026-08-18 同步）。

## 1. 这个项目是什么

第二篇方法论文的代码与实验仓库——系统刻画 **DZ-Unmix 方法的适用边界**。延续 [`../dz-unmix-cg/`](../dz-unmix-cg/) 第一篇 C&G 论文，但**独立目录**。

核心 research questions 见 [README §1](README.md#1-项目目标)。

## 2. 硬约定（不要违反）

### 🚫 不要修改 `../dz-unmix-cg/`

- 第一篇论文代码 **DOI reserved**（10.5281/zenodo.20371892，⚠ 未 publish、当前 404——见 master §3 HIGH#1）；论文本身已投 C&G（CAGEO-D-26-00963，R1 已交 2026-08-11、等 2nd decision）
- 要复用其代码：通过 `src/adapter_cg.py` 薄包装引入，**禁止改原文件**
- 即使发现 bug，先和用户确认是否影响第一篇论文，再决定 fix 在哪里

### Stub-first 开发顺序

v0 骨架阶段 `src/runner.py` 调用 `_stub_run()` 返回 mock 指标——**先让整条 pipeline 跑通**，再接真实方法。Mock 故意符合定性预期（accuracy 随 K↑、N↓、噪声↑、相似度↑ 而下降）。

升级到 v1 时不要直接删 `_stub_run`，加 `--stub` flag 切换，方便回归测试。

### 配置驱动，避免硬编码

- 参数维度全在 `config/*.yaml`——加新维度改 YAML，不改 `src/grid.py`
- `src/grid.py` 的 `_expand_values_dict` 自动忽略 description-only 字段，YAML 可以自由加文档

### Idempotency

- `runner.py` 检查 `outputs/{grid}/results/{config_id}.parquet` 存在则跳过
- SLURM 重投只会跑未完成 config——这个性质不要破坏

## 3. 与 `dz-unmix-cg` 的边界

| 维度 | dz-unmix-cg | dz-unmix-bounds |
|---|---|---|
| 论文 | 第一篇（C&G · **CAGEO-D-26-00963**，R1 已交 2026-08-11、等 2nd decision）| 第二篇（适用边界，**C&G 07-31 desk reject → 2026-08-05 已投 MDPI Minerals（minerals-4514253）**，08-07 疑桌面拒待核邮箱；C&G 前史稿号 CAGEO-D-26-01238。07-07 旧注＝当时"已投"系误记；Zenodo concept 10.5281/zenodo.20768131，当前版本 v1.1 = 21739169）|
| 状态 | 冻结，DOI reserved（未 publish） | ⚠ 本地目录＝过期 v0 stub 存档；实际开发已移至 GitHub 仓（v1.1）与权威稿件目录 |
| 扫描维度 | seed × α × bandwidth（75）| + K, N, ρ, ε（数千到数万）|
| 方法 | Supervised NMF 单一 | + DZmix / Sambridge（v1 计划）|
| 数据 | 合成 + NE 亚 K1 4 盆地 | 合成为主 + 真实数据回测（v2 计划）|
| 计算 | 单核串行 | SLURM job array（D107）|

## 4. 当前 v0 骨架的范围与限制（⚠ 历史存档：v1/v2 从未在本目录落地——论文实际实验/图在 `../sigma_err_gpu/`、第二篇稿件目录与 GitHub 仓完成）

**v0 包含**：grid expansion、stateless task table、stub runner、SLURM 提交模板、smoke test

**v0 暂不包含**（明确 out of scope）：
- 真实方法接入（adapter_cg.py 是 stub）→ v1
- 结果聚合脚本（slurm/aggregate.sh）→ v1
- 第二个方法 plugin（DZmix wrapper）→ v1
- 真实 DZ 数据集回测 → v2
- 论文图表生成 → v2

如果用户提了 v1/v2 范围内的事，**问一句确认是否要扩展 v0 scope**，不要默认展开。

## 5. 反 scope creep 提示（继承用户全局约定）

- 做完用户明确要求的事就停
- 想到额外可做的：1-2 句话提，让用户选
- 不主动做元工作（项目重构、依赖升级、cleanup）
- 用户说停立刻停

## 6. 环境备忘

- Windows 11 主机 + D107 集群（SLURM）
- 主机 Python：~~`C:/Users/Administrator/miniconda3/python.exe`~~（⚠ 已换机 Administrator→Lenovo，旧路径失效；现用 `C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`，numpy/pandas/pyarrow/matplotlib/scipy 齐）
- 中文文件 PowerShell 编码陷阱见用户级 `~/.claude/CLAUDE.md` §6
- OneDrive 同步偶尔 file lock，写失败**重试一次**通常就过

## 7. 重要文件入口

| 文件 | 用途 |
|---|---|
| `README.md` | 项目目标、设计决策、跑法 |
| `config/grid_pilot.yaml` | 维度定义 + 文档化的 schema 范本 |
| `src/grid.py` | YAML → task table 展开器 |
| `src/runner.py` | 单 config 执行器（v0 stub）|
| `slurm/submit.sh` | SLURM job array 模板 |
| `tests/test_dry_run.py` | smoke test，pytest 直接跑 |
