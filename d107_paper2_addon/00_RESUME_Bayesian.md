# E 线 · Bayesian DZ-Unmix 不确定性量化 · RESUME（续工先读这个）

> 建于 2026-07-04（应"给 E 线也起个 RESUME"）。E 线 = `00_论文总览_MASTER.md` 的 **E 线**（内部旧称"第三篇"，⚠ 与目录名 `d107_paper2_addon` 的 "paper2" 冲突——见文末命名雷区）。
> **详细真相源 = 同目录 `PAPER3_bayes_materials.md`**（Table 1 + 全部图注 + Intro/Methods/Results/Discussion 草稿 + M3 三步留痕 §0.1–0.3 + gap 清单 §6）。本 RESUME 只做冷启动索引，不重复搬运。

> ## ✅ 2026-10-13 状态回填：Approve 已完成
> **正式稿号 = CAGEO-D-26-01540**（✔ Downloads `2026-09-08 2333 Submission to Computers and Geosciences - manuscript n.eml`：Manuscript Number CAGEO-D-26-01540；导航 Submissions Being Processed）。此前 Incomplete / 待 Approve 叙述作废。现被动等编辑部处理 / 外审。MASTER §1-E/§2-E 同日已同步。
>

## 这是什么

把三端元碎屑锆石物源解混做成**贝叶斯反演 + 全后验不确定性量化**，应用到 NE 亚 15 个下白垩（K1）真实样本。核心方法学卖点 = **GPU 批处理采样器**（181,500 fits，74 min vs CPU emcee 数天；逐拟合与 emcee 等价，|diff| 0.020 ≪ emcee 换种子自身抖动 0.19）+ **逐权重可信度分级**（不是所有权重同等可信）。

- **端元**：S1 Mesozoic arc（~130 Ma）/ S2 CAOB basement（~280 Ma）/ S3 Archaean–Proterozoic craton（~1900 Ma）。
- **样本**：4 NCC north-margin + 11 Songliao（OneDZ/Puetz，n=31–111），坐标已入 Table 1。
- **2026-07-04 用户拍板**：正式立**独立稿**（此前寄生在 bounds 仓 addon）。M3 作为增强章节挂靠。

## 现在到哪了

- **数据侧：闭合**。181,500-fit GPU scan 全量 = 本目录 `grains_bayes_gpu.parquet`（40MB，reviewer 复算真相源）。主结果表 Table 1（15 样本，逐权重 mean±std + Resolvability）成形（素材包 §1）。
- **M3 增强分析：三步全跑完（2026-07-04）**，是本轮最重的进展（留痕素材包 §0.1–0.3）：
  1. 后验协方差特征分解 → 软方向沿 S1–S2 边（11/15 样本 <2°）；**新发现"改道"**（arc 缺席样本软方向转 S2–S3）；
  2. n-标度干净检验 → slope **−0.507**（预测 −0.5），且软度 = **计数噪声地板 ×1.02** ——**不是端元相邻弱可识别**（130/280 Ma 数据层面 KS≈0.999 可分）；
  3. within-vs-between 方差分解 → 单次 credible 已占总方差 ~一半（扫描只抬 ~1.4×）、**先验近乎惰性（<3% between 方差，13/15）**。
- **写作侧：四节草稿齐（2026-07-04，⚠ 比 master §2-E "正文 0 字" 新）**。Intro + Geological setting（素材包 §2.5）、Methods（§3）、Results（§4，**已按 M3 三步重写**，修正了原稿"端元相邻→软"的幅度误归因）、Discussion（§5.1–5.6）。**学术声音 + 占位引用待本人定稿**。
- **图/表：Fig 1–6 全部本地渲染 + Read/auditor 视觉验证（2026-07-04）**。Fig 1 位置图（`Fig1_sample_locations_v1.py`，geo-figures auditor 0 issues，真实经纬度）/ Fig 2 UQ 三元图+dot-whisker（`fig_bayes_uncertainty`，parquet 路径已改本地、caption 按 M3 对齐、修了标签重叠）/ Fig 3 方差来源堆叠条 / **Fig 4 监督 NMF vs 贝叶斯对照**（`FigY_nmf_comparison.py`，限 4 NCC 重叠样本，MAE 0.03；= 原 P0 硬缺口）/ Fig 5 μ/σ 后验 vs 先验 / Fig 6 acceptance 诊断。**表**：Table 1（逐权重 Resolvability + lat/lon）+ Table 2（bootstrap 收敛）。

## 下一步（P0 / P1 / 引用 / 打包 均已完成 2026-07-04；剩下多为本人定稿动作）

1. **✅ P0 监督 NMF 对照图（Fig 4）已做**——`FigY_nmf_comparison.py`：把第一篇 `supervised_unmix`（只读调用）应用到 4 个真实 NCC 样本，对贝叶斯后验均值±std，**MAE 0.03**（硬 w_S3 0.01 / 软 0.04，点估计落误差棒内）。⚠ **审计修正**：第一篇 case study 用的是**文献参数化合成盆地代理**、并没跑这 4 个真实样本 → 这是"应用第一篇方法到真实数据"，非"取第一篇结果"（§5.4 / §6 已注）。对照仍限 4 NCC（11 松辽第一篇没跑）。
2. **✅ P1 补图/表均已做**：Fig 5 μ/σ 后验 vs 先验（数据实质更新形状：μ 收窄 0.31–0.53×、σ_S2/S3 右移；先验惰性）、Fig 6 acceptance 诊断（99.3% 在 [0.2,0.5]、median 0.42）、Table 2 bootstrap 收敛（b≈20 中位比值≈1.0 → 100 是余量非过度）。
3. **✅ 引用补全 + DOI 全核实**：§2.5 地质/方法 8 条（Zhao 2012/2001、Liu 2017、Xiao 2014、Huang 2024、Li 2021 + Sharman-Johnstone/Sundell-Saylor）+ M1 companion（第二篇，**已投 C&G**，代码 DOI 10.5281/zenodo.20768132）+ 样本源 DOI——**逐条 Crossref 交叉核、卷/页齐**。
4. **✅ Zenodo 包已备好**：`zenodo_paper3/package/`（52 文件；精确筛 E 线、排除 100+ 个 M1/M2 `probe_*` clutter；README/MANIFEST/data_availability/requirements/assemble.ps1 齐）。
5. **目标刊 = *Computers & Geosciences***（2026-07-04 定，与第二/四篇同刊）。

**真正剩下的（本人动作为主）**：
- **学术声音定稿**：Intro/Methods/Results/Discussion 是草稿，按 C&G 模板 + 你的声音改定。
- **✅ §5.5 SL0932 层位判断已给（2026-07-04 本人）→ 已写入**：泉头组记录**局部基底隆起就近供源**；保留 hard-result 强调（w_S3=0.87±0.034 是硬权重、非软方向伪影）；层位支撑 = SL0932 是 SL09xx 簇唯一泉头组样本（余 SL0929/30/31 = 登娄库组）。
- **LICENSE（源仓库无，建议 MIT+CC-BY-4.0）+ 上传 Zenodo 铸 DOI 回填** + C&G 系统填 CRediT/Declaration/日期 → submit。
- **M1 稿号回填 §5.4**：第二篇已投 C&G，拿到稿号（可能 = CAGEO-S-26-01308，若与「方法稿」同篇）后回填。

## 定位纪律（写作硬约束）

- **双向引用不重证**：与第二篇（bounds）M1 可识别性定理是**解析(方向) vs 经验(幅度)**分工——M1 给软方向解析刻画（弱特征向量 (1,−1,0)/√2），本文给有限-n 幅度；正文一句"解析刻画见 [bounds 第二篇]"即可，别重推定理。与第一篇（dz-unmix-cg）同场景同端元，引为端元/方法出处。
- **卖点收窄**：核心是"逐权重可信度 + 计数噪声地板可操作采样建议"，不是"贝叶斯首创物源反演"。M3 重写后叙事升级为**逐样本判定"不可约有限-n 地板 vs 端元形状灵活性膨胀"**（前者加 grain 能救，后者不能）——比原稿更利审稿。
- **self-scoop**：本线已在 A（方法稿）、B（bounds）两篇 cover letter 里作为 in-prep companion 披露；数字首发留本线。
- **数据 license**：Puetz(R 系列)/OneDZ(SL09xx) 来源引用 + license 沿用 `dz-global-db` 已核结论，别重核。

## 环境备忘

- **全量 parquet 在本目录** `d107_paper2_addon/grains_bayes_gpu.parquet`（40MB），**不在** `Projects/bayes_gpu/`（那是代码工作目录）。
- **fig 脚本路径**：`fig_bayes_uncertainty.py` 原写死 L20 容器路径，**已改为 `Path(__file__).resolve().parent / grains_bayes_gpu.parquet`（可移植，本地已渲染验证）**；其余 `m3_*` / `Fig1_*` / `FigY_*` 脚本均用脚本同目录/相对路径，本机直接可跑。
- **本机 Python**：within-vs-between 那步在 Lenovo 机跑，用 `C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`（winget 装，numpy/pandas/pyarrow 齐；本会话又补了 matplotlib/scipy）。前两步在 Administrator/D107，产物经 OneDrive 同步。M3 脚本 `m3_*.py`（区别于 M1 的 `probe2_*`）在本目录。
- **图改完必须渲染 + Read 视觉验证**（全局约定）。

## ⚠ 命名 / 目录雷区

- **三重命名冲突**：本线 master 叫 **E 线**、素材包内部叫 **"第三篇"**、寄生目录名叫 **`d107_paper2_addon`**（"paper2" 指贝叶斯 addon 物理挂 bounds 仓，**不是**第二篇）。**Zenodo/正文命名别被目录名误标成第二篇。**
- **目录整合是选项、非本次动作**：素材（本目录）+ 代码（`Projects/bayes_gpu`）+ 全量 parquet（本目录）分散。若要像第四篇那样立独立目录（如 `Projects/第X篇_Bayesian物源反演/`）需搬迁 + 定编号——**涉及 §编号轴决策（master §4）+ 跨仓搬迁，按约定先与本人确认再动**，本 RESUME 未做。
- **同源经费**：与 2026 石家庄市科技计划申报"沉积大数据物源贝叶斯反演与不确定性量化"主题一致，很可能就是为这条线申的经费。

## ⚡ 2026-07-04（晚）投稿收尾轮 — 做到哪了

**用户"E 线也开始收尾投稿"→ 大幅推进，现卡在一个 Zenodo 手动开关。**

**✅ 本轮完成**：
- **全文组装** `08_manuscript_full.md`：把素材包 §2.5/§3/§4/§5 + Table 1/2 + 6 图注 + Highlights + Graphical abstract + Abstract + Conclusions + CRediT + Declarations + Data/Code availability + References 组装成单文件完整投稿稿（标题定为 "Which mixing weights can you trust? GPU-accelerated Bayesian uncertainty quantification..."）。备份 `.preassembly_polish_bak` / `.postpolish_bak`。
- **Fable polish（129-agent workflow，53 收 9 拒）**：linter 全清（7 because / 7 but / 1 though → 0；relative which 9 处 → 0，含补掉一条 verify 拒但为全稿一致仍改的 pied-piped）+ 术语统一（counting-noise floor / subsampling / GPU-versus-emcee / end-members / scan std）+ artefact UK 拼写 + Highlight #4 压到 ≤85 字符。**红线审计：number multiset 955=955 IDENTICAL**、banned 0、artifact 0。changelog 见本 RESUME 下方 + workflow 结果留痕。
- **引用补全**：References 补 Foreman-Mackey 2013 / Goodman-Weare 2010（emcee/GW 采样器，Crossref 已核卷页）；companion 引用 = Pan submitted-a（方法稿，SSRN 6955308 + dz-unmix-cg）+ Pan submitted-b（bounds，Zenodo 10.5281/zenodo.20768132）。**用户拍板 [9]/companion 不带稿号引**（学术惯例够用；CLAUDE.md 两行"是否同篇"合并问题**仍待你定**，不阻塞本稿）。
- **GitHub 仓已建并 public**：**https://github.com/PNOAHA/dz-bayes-uq**（59 文件，用户授权后 push，commit 见 git log）= Zenodo 包内容 + 新写 LICENSE(MIT)/LICENSE_DATA.md(CC-BY-4.0)/CITATION.cff/README/.gitignore。本地仓 `Projects/_repo_dz-bayes-uq/`。
- **cover letter** `09_cover_letter_bayes.md`（含 self-scoop 声明、companion 关系、DOI 占位）。
- **目标刊确认 = C&G**（用户 2026-07-04 再确认，三篇同投不换刷）。

**✅ 2026-07-07 Zenodo DOI 已铸 + 回填（原 blocker 解除）**：
- 用户翻 Zenodo↔GitHub sync 开关 ON（`PNOAHA/dz-bayes-uq`）→ LN `gh release create v1.0.0`（proxy 会漂移，当日实测 5556；gh `C:\Program Files\GitHub CLI\gh.exe` 认证 PNOAHA）→ Zenodo 自动铸并 **published**：**concept DOI `10.5281/zenodo.21242245`（进正文）** / version DOI `21242246`。元数据一手核过全对（title/author Pan Lujia+河北师大数院/MIT/Software/isSupplementTo v1.0.0，无需后台改）。
- **已回填**：`08_manuscript_full.md` Data availability（结构化 Availability 字段）+ `09_cover_letter_bayes.md`（DOI 那行）。**cover letter `[Submission date]`（L7）留投稿当天填。**
- **仓库 README/CITATION.cff 加 DOI badge = 可选未做**（要另一次 push，公开动作；非投稿必需）。

**⏸️ 2026-07-07 投稿前审查（3-lens workflow）+ 已做的机械修**：
- **已修（机械层，基线备份 `08_manuscript_full.prereview0707_bak.md`）**：加 "Declaration of generative AI" 节（Elsevier 强制）；关键词 8→6（删 MCMC+North China Craton）；Data and code availability → C&G 结构化字段；"bit-reproducible"→"reproducible"。
- **✅ 5 个科学/一致性点已全部处理（2026-07-08，用户拍板"2/3/4 按建议做，5 先投 B"）**：
  ① **05FW008**：对称处理——"cleanly separates"→"largely separates"（Abstract/§5.5/结论6），§4.3+§5.5 明写"4 个 NCC 里 3 个有 craton 信号；05FW008 arc 主导（w_S3=0.026）= 松辽例外 SL0932 的 NCC 对侧对应"。**未编地质原因**（用户没给；如有近源富弧层位解释可后补一句）。
  ② **"<3%" 判定为正确非 bug**（分母=between-fit variance，正文 §4.2/§5.3/结论5 本就带分母；审查 agent 误判）——修法=给缺分母的 3 处（Abstract/Highlight/GA）补"of the between-fit variance"，数字不动；摘要顺带压回 298 词（≤300）。
  ③ **GPU 等价补 CI 宽度**：§3.4+Abstract 补"posterior std 一致到同一 0.02 容差→复现的是可信区间宽度不只均值"（据 validate_gpu.py PASS 门槛 tol_std=0.02，未编具体值）；"several days"→"~44 CPU-core-days"（21s×181500，算术自洽）。
  ④ **DZmix 定位**：§1 加 2 句——承认 Sundell&Saylor bootstrap CI，点明三点延伸（全后验+先验扫描/逐权重分级/分离计数噪声地板）。
  ⑤ **submitted-b 保留不动**，投稿顺序=**先投 B 再投 E** 坐实。
  另：Li et al. 2021 References 条目 Crossref 核实修正（全作者 Li, H., Han, J., Yin, Z. 仅 3 人；标题照发表原文 "Paleo-Pacific ocean"）。
- **✅ 数字审计 PASS**（基线 `prereview0707_bak` vs 现稿：零丢失，13 个新增 token 逐个对账合法，DOI 占位→21242245）；house-style 复扫 0 违规；稿内占位符 0。
- **✅ 投稿件已生成**：`submission_package/manuscript_bayes_uq.docx`（pandoc 3.9，Word 计数 7296 词；equations=0 属正常——E 全稿用 Unicode 记号无 LaTeX 数学）。**Graphical abstract 图像 = 直接用 `fig_bayes_uncertainty.png`（Fig 2）**，GA 文字描述本就是"built from Fig. 2"。
- **剩**：cover letter `[Submission date]`（L7）投稿当天填 + 生成 cover letter docx（或 EM 粘文本）→ **等 B 投出后**用户登 C&G EM 投 E。

## ✅✅✅ 2026-07-08 20:07 已投 Computers & Geosciences（终态）

- **确认页坐实**：Submitted 20:07, July 8, 2026；Research Paper；新标题 "Grading the credibility of detrital-zircon mixing weights: GPU-accelerated Bayesian uncertainty quantification of Lower Cretaceous samples from NE Asia"；单作者 Lujia Pan/corresponding。列表页状态 "Submitted"。**正式稿号 CAGEO-D-26-xxxxx 待确认邮件 → 回填此处 + master + memory。**
- 投稿包终版：manuscript_bayes_uq.docx(17:33,新标题) + Fig1(18:02 去 FGDC 水印) + Fig2-5(15:45) + Fig6(17:49 去图级标题) + GA(Fig2 png) + highlights_bayes.docx + cover_letter_bayes.docx(19:49,门户 required 有独立槽位,全交)。
- 投前当日全链：R5 polish(1 处 to within)+ 换标题(避开与 B 同构)+ References 13/13 Crossref 外部核验零伪引 + 六图逐张视检(Fig6 去图名/Fig1 去水印)。
- 当日双杀:B 15:40 + E 20:07,同刊同日两篇。**剩 F 未投(六线最后一篇)。**
- 待办(非阻塞):repo CITATION.cff preferred-citation 改新标题+补稿号,连同 README DOI badge 一次 push。

## 📌 2026-09-01 补记：若审稿人问 IAT/收敛，怎么答（届时触发，现在不跑）

> ⚠ **2026-09-02 更正**：07-15 Sent Back 信**已读**（原件 `EM_SentBack_2026-07-15_CAGEO_E_line_numbers.eml`），原因仅 **Line numbers**。真实状态 = 07-19 已点 Edit Submission 但未走完 → 稿件一直停在 **Incomplete Submissions**（不是「未拆」）。09-01/02 三轮 polish + 重建：`submission_package/*_v3.docx`（带行号）+ 六图重编号 = 终态，**待 EM 替换文件 → Build PDF → Approve**。全程留痕 `POLISH_LOG_E.md`。

### ⏩ 续工入口（2026-09-02 写给下一个 session · 先读这段）

**状态一句话（2026-10-13 更正）**：E 稿 **09-08 已 Approve**，正式稿号 **CAGEO-D-26-01540**（曾 Being Processed）。此前 Incomplete / 待 Approve 已过时。退回信原件 `EM_SentBack_2026-07-15_CAGEO_E_line_numbers.eml`，原因仅 Line numbers；三轮 polish + v3 为 Approve 前终态。

**📌 09-02 12:15 补记（续工 session 预检）**：只读交叉核全过（docx 行号/图序/Highlights 字符数/六图文本层，详 `POLISH_LOG_E.md` 末段）。**已改 1 处**：cover letter 日期 → 2 September 2026，`cover_letter_bayes_v3.docx` 已重建（12:14），稿件/Highlights v3 未动；非当日上传须再改日期。**✅ 12:20 用户拍板 (c) 已执行**：B 线 08-07 已遭 Minerals 桌面拒且未再投（.eml 在 Downloads），E 稿 §5.4 / References / cover letter 三处 "submitted-b / submitted elsewhere" 已改 "in preparation"，三件 v3 docx 重建（12:20）并双路复核（逐段 diff 仅预期 3 段变化 + Word COM 行号 Active），数字 multiset IDENTICAL。**投稿包终态，可直接按下表上传。** 随后用户拍板去掉 `submitted-a` 后缀：6 处 → `(Pan, submitted)`，三件 v3 再重建并同法复核（稿件恰 6 段变化、行号 Active）。companion 引用终态 A = (Pan, submitted) / B = (Pan, in preparation)。

**✅ 09-02 17:50 用户拍板"都按建议改"→ 全部执行完毕，投稿包重回终态（三件 v3 docx 17:5x 重建，上传清单不变）**：① References 13→**39**（新增 24 条 must+should，Ansel 2024 因 49 作者与 numpy/scipy 等未引软件不一致而未加，改在 Software required 写 torch；另加 FGDC 2006 / Okabe & Ito 2008 两条灰色文献）+ Liu 2017 两错订正 + Li 2021 / Xiao 2014 补期号 + B 线 Zenodo 两处改 concept 20768131；② 四处表述修正（§1 bootstrap→resampling + Tipton 2022 先例；Solonker 加 Xiao 2003；Data availability 数据库正名 OneDZ = Li, K. et al. 2026 / Puetz et al. 2024；Table 1 注与 Fig 1 注 dz-global-db→OneDZ/Puetz）；③ **K1 措辞改为 Jurassic–Cretaceous**（标题/摘要/§1/§5/结论/Table 1 题/Fig 1,3,4 注/GA 注/cover letter），§2 新增按组层位句（依据：OneDZ 原始行 163–100 Ma 窗、Xu 2016 摘要、西山层序检索、**归档 CSV 颗粒年龄：05FW008 全部 36 颗 142–164 Ma → 沉积在 J/K 界线附近；05FW007 除一颗 143 外无 <180 Ma 颗粒 → 中侏罗世可成立**），Table 1 加 Formation 列，S1 端元条与 §5.5 各加一句"对 05FW007/008 S1 代指更老的侏罗纪弧锆石"提醒，Quantou "basement high" 改为两解释并列（Li 2012）且 Conclusion 6 同步；④ 4 路对抗 QA（引用完整性/句子级/层位一致性/数字不变性）15 条全部修入，数字不变性零问题；⑤ **E 线 Zenodo v1.0.1 已发**：GitHub release https://github.com/PNOAHA/dz-bayes-uq/releases/tag/v1.0.1（commit 8d5e4a5），Zenodo 版本记录 **10.5281/zenodo.22251075**（concept 21242245 现指向 v1.0.1；稿件引 concept 不用改）。备份链：`pre0902prep` → `pre0902suffix` → `pre0902refs` → `pre0902qa`。 **09-03 09:37 补：表格三线补丁**（pandoc 默认表无顶/底线，构建脚本新增 `patch_table_borders()`，Word COM 实测两表 top/bottom/header 线齐全），三件 v3 重建为 09-03 09:37 版，文字未变；上传清单不变。 **10:07 再补 Table 1 横版补丁**（EM Build PDF 把 10 等宽列硬排、样本号拆行跨 3 页；现为竖–横–竖三节 + 显式列宽 9.0 in + 9 pt，Word COM 实测 Table 1 落在单页），三件 v3 = 09-03 10:07 版；**10:20 再补两件**：Fig 1 图例改字重渲染（去 'K1 DZ samples' / 'Wide-rift / back-arc'）、GA 改矢量 PDF `GraphicalAbstract_Fig3.pdf`（EM 把 PNG 压成 72 dpi；Fig 1–6 本就是矢量嵌入不糊）。**EM 需替换三件：manuscript_bayes_uq_v3.docx（10:07）+ Fig1_sample_locations.pdf（10:16）+ Graphical abstract 换 PDF** → 重新 Build PDF → 看 Table 1 横版页与 Fig 1 图例 → Approve。仓库 Fig1 改动已本地 commit 未 push（v1.0.2 待拍板）。 **10:45 第二版 EM PDF 全过，仅 n 列 111 折行 → 列宽 0.45 重建（稿件 = 09-03 10:45 版），EM 再换一次稿件 → Build → Approve。** **10:55 第三版 EM PDF（CAGEO-S-26-01685 (2).pdf）全项通过：Table 1 单横版页 n=111 完整、行号 1–749、GA/六图矢量、39 条文献 → 已告知用户可 Approve；Approve 后回填稿号。**⚠ 用户仍宜自读一遍 §2 层位句与 §5.5 括号句（唯一由我代作的科学表述）。

**文件**：稿件母本 `08_manuscript_full.md`（数字红线基线 `08_manuscript_full.prepolish0901_bak.md` 974 token → 现 967，23 项受控差异全在 `POLISH_LOG_E.md`）；投稿包 `submission_package/`：`manuscript_bayes_uq_v3.docx`（带行号，Word COM 实测 Active）/ `highlights_bayes_v3.docx` / `cover_letter_bayes_v3.docx` + `figures/` Fig1（未动）Fig2–6（按首引重编号 6→2,2→3,3→4,4→6；Fig5/6 重渲染）+ GA png（未动，文件名 _Fig2 是旧号、内容=新 Fig3，不重传）。旧件：7 月三件原 docx 保留；`figures/_superseded_0901/`。**改 md 后重建**：`python build_e_resubmission.py`（纯 pandoc + lnNumType 补丁 + 段落探针对账）。

**EM 上传清单**（Main Menu → Incomplete Submissions → Action Links → Edit Submission → Attach Files；🔴 全程勿碰 Remove Submission）：
| 条目 | 删 | 传 |
|---|---|---|
| Cover letter | cover_letter_bayes.docx | cover_letter_bayes_v3.docx |
| Highlights | highlights_bayes.docx | highlights_bayes_v3.docx |
| Manuscript | manuscript_bayes_uq.docx | manuscript_bayes_uq_v3.docx |
| Figure ×5 | 旧 Fig2–Fig6 | Fig2_acceptance_diagnostics / Fig3_bayes_uncertainty / Fig4_variance_sources / Fig5_mu_sigma_posterior / Fig6_nmf_comparison（Description 填文件名） |
Fig1 / Graphical abstract / Declaration 不动；Order 保持 Cover→Declaration→Highlights→GA→Manuscript→Fig1…6。Proceed → **Build PDF for my Approval** → Waiting for Author's Approval → View Submission **逐页看行号 + 图序 1–6** → **Approve Submission**（只 Build 不 Approve = 没交，07-19 就卡在这）。

**可选未做（非阻塞，用户拍板）**：① `## Data and code availability` 节名是否改 "Computer Code Availability"（B 用 "Code and data availability" 过了预审，判非必需）；② §5.4 "the methods paper" 指 submitted-b 略歧义（可改 "the bounds paper"）；③ "thirteen/fifteen" 数词混用未统一；④ `zenodo_paper3/package` 内旧图/旧标签脚本与新编号未同步（已发布 record 改不了，接收后走 New version）；⑤ repo CITATION.cff 标题/稿号顺带 push。

**Approve 之后**：回填 MASTER §1-E 状态与 memory `cageo-two-papers-sent-back`；稿号（CAGEO-D-26-xxxxx）从确认邮件回填。审稿若问 IAT/收敛 → 见下文 📌 段。
> 本段来自第五轮 GPU brainstorm 对抗审查的合法残余物：**预跑被判负期望（KILL），只留触发式预案。**

Fig 6 图注自认 per-fit IAT 未存储。**当且仅当**审稿意见真的问到 IAT/ESS/链收敛时，跑：

```
cd /private/bayes_gpu && nohup python run_scan_gpu.py --n-steps 20000 --burn-in 4000 --out-dir /private/bayes_gpu/out_iat > out_iat.log 2>&1 &
```

- 种子全确定（bootstrap=sha1(name,b)、MCMC seed=20260609 硬编码），同 seed 加长链是纯延伸；其余参数吃默认=与发表版口径一致（64 walkers/thin 4/float64/11×11×100）
- 成本 ≈ 4×74min 单卡 ≈ 5 卡时；跑完与 `out/grains_bayes_gpu.parquet` 对账 max|Δweights_mean| 进 response letter
- ⚠ **不要提前跑**：结果不变则对账表零信息；结果会变则是在给在审稿制造披露义务（红线④）。IAT 计算需在 bayes_gpu.py 存链后加（现版 thinned 链在 `run_mcmc_gpu` 内即弃只留矩），届时改 ~10 行

## 一句话给新 session

**⚠ 2026-07-08 投稿当口改标题**（用户点出与 B 同构"问句+副题"套模板观感差）：**新标题 = "Grading the credibility of detrital-zircon mixing weights: GPU-accelerated Bayesian uncertainty quantification of Lower Cretaceous samples from NE Asia"**（动名词开头、贡献前置；正文+cover letter+docx 三处已同步,docx 17:33 终版）。旧问句标题作废;repo CITATION.cff 的 preferred-citation 标题待日后顺带 push 更新（非阻塞）。

**E 已 100% 投稿就绪且已解锁**：全文终稿（科学 5 点全处理+审计 PASS）+ docx 已出 + Zenodo DOI 21242245 已铸回填 + cover letter 齐（只差投稿日期）。**✅ B 已于 2026-07-08 15:40 投出 C&G（portal ID 584662dc-…）→ E 的 "submitted-b" 及 cover letter "two companion studies I have submitted" 均已为真，E 随时可投**（走 submit.elsevier.com/CAGEO 同一门户，流程参照 B：分类建议 Geochronology/Inverse Problem/HPC/Statistics/Sedimentology 同款、research data=Zenodo 21242245、SSRN 选 YES）。目标刊 C&G。
