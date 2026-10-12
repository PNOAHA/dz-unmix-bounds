# E 线稿 polish changelog（2026-09-01 起 · 用户要求多轮全面 polish）

> 对象 = `08_manuscript_full.md`（docx 母本）。基线 = `08_manuscript_full.prepolish0901_bak.md`。
> 纪律：数字 multiset 每轮 IDENTICAL；judgment 级一律先过用户；每轮零错即停判据 = proofread lens。
> ⚠ 全部轮次结束后须：重建 docx → 重打 lnNumType 行号补丁 → 与 md 对账 → 才能传 EM。

## Round 1（2026-09-01，wf_70b873c8-3a0：8 lens + 3 对抗验证者；60 findings→29 low + 26 judgment CONFIRMED，3 REJECTED）

### 已应用（22 处 low 级，数字 multiset 974 token IDENTICAL）

| 类 | 处数 | 内容 |
|---|---|---|
| Tier 4H 弱动词 | 4 | L92/348/390/708 现在时 show(s) that → demonstrate/reveals/demonstrates |
| Oxford comma | 4 | L236/701/752 三项并列 + L235 斜杠断行合并 |
| %-空格统一 | 8 | 图注区 "10 %"→"10%" 系（L723-770，11 个实例），比较算符 "<3%"→"< 3%"（L15/52） |
| 符号/格式一致 | 6 | ↔ 去空格×2、Table 2 标题加粗范围、Okabe–Ito en dash、magenta→reddish-purple、其余 |

REJECTED 3 条全部是"改数字 token 版本"被验证者拦下（0.05→0.03 两处、σ 单位第二处连带图轴问题）——同问题的合规版本进入 judgment 清单。

### 待用户拍板（judgment 级，fix 已备好未应用）

见本轮汇报消息；最重 = "< 3% of between-fit in 13 of 15" 家族与 CSV 重算矛盾（仅 8/15 成立）。

### 验证者层拦截记录（含金量样本）

- [35]/[36] 0.05→0.03：issue 属实但 fix 违反数字红线 → REJECTED，"typically" 版胜出
- [39] σ 单位改注：Fig5 图轴本身印着 "peak width, Ma"，只改注会造成图-注新矛盾 → REJECTED，须图文联动决策
- Fig5 疑似 typo "pripr" 经 pypdf 文本层证实为 "prior" 渲染伪影 → 假阳性未报

### Round 1 追加（2026-09-01 用户拍板：C、D 全收）

C 组 10 处 + D 组 7 处 = **17 处已应用**。multiset 受控差异恰为预期 4 项：
`0.02→0.020`（用户批准）、`§1` 的 1 与 `Fig. 3` 的 3 各 −1（删 Table 1 caption 内部备注）。其余 970 token 不变。

### A 组重算结果（2026-09-01 LN 独立复算，csv=m3_within_vs_between_results.csv）

- **"prior < 3% of between-fit in 13 of 15" 证伪坐实**：三权重方差加总口径 **8/15**、逐权重中位口径 **8/15**、frac_boot≥0.97 口径 **8/15**——三种合理聚合全部 8/15。
- 超标七样本：S12652 45.6% / 05FW007 41.7% / **05FW008 33.2%（稿件从未点名的第三个大户）** / SL0929 12.0% / S12650 6.5% / SL0930 6.1% / SL0931 3.2%。
- **"a median of under 3%" 为真**（逐样本 median 1.8%，45 权重 median 1.4%）。
- of-total 口径（Fig 3 注）：median 0.85%≈图注 0.8% ✓；"≤6% 13/15" 在我的 total² 加总口径下是 **12/15**（第 13 个卡在 6.04% 边界）——若改用此口径须先用图脚本原始聚合复核。
- 结论：错的是"13 of 15"这个计数（疑似从 of-total 口径串进 between-fit 句），"median<3%"保留可用。待用户选改写方向后一次改齐全部平行副本（Highlight/GA/Abstract/§4.2/§5.3/Conclusions/Table 1 注）。

## Round 1 · B 组执行记录（2026-09-01，用户拍板 A=方案iii + B1-B5 全做）

**A（方案 iii）**：8 处改写全落地——Highlight/GA/Abstract/§4.2/§5.3×3/Conclusions/Table 1 注/Fig 3 注。
统一口径 = 保留为真的 "median < 3% of between-fit"，删全部 "13 of 15" 计数，§4.2 与 Table 1 注点名三例外
（05FW007/05FW008/R31370-S12652），§5.3 "only two"→"contributes most"、"essentially all"→"most"。
Fig 3 注的 of-total 口径删 "≤6% for 13/15"（12 vs 13 边界未坐实）保留 median 0.8% + 两极端。
token 受控差异 9 项与预告逐项一致（13/15 各−6、97−2、6−1；三样本名 digits +8）。
⚠ §5.6 的 "n^(−1/2) law for thirteen of fifteen" **未动**——那是 floor 口径（15−2 shape-inflated），与 prior 家族无关。

**B1 重编号**：old→new = 6→2（acceptance diagnostics）、2→3（bayes uncertainty）、3→4（variance sources）、
4→6（nmf comparison）；1、5 不动。占位符两步置换防串号；正文首引严格 1→6；文末图注物理重排；
PDF 已按新号改名（旧版留 `submission_package/figures/_superseded_0901/`）。GA 描述自动变 "built from Fig. 3"（正确）。

**B2 GA**：left/right→top/bottom，"annotated with"→"Two facts govern the picture"（PNG 未动，无需重传 GA）。
**B3**：新 Fig 6 轴标签 "(first paper)/(this paper)"→"(companion study)/(this study)"，重渲染+视检 ✓。
**B4**：Fig 5 σ 轴 "peak width, Ma"→"Myr"（μ 轴 Ma 不动），图注两处同步，重渲染+视检 ✓。
**B5**：Fig 3 注删 "in every sample"（05FW008=87.5%）；Fig 6 注 "every sample's median sits" ✓。

**投稿前重建链（未做）**：md→docx（pandoc）→ lnNumType 行号补丁 → 与 md 对账 → EM 替换稿件 + 4 张改名图。

### Table 2 重生成（2026-09-01，用户批准）

诊断：13 种 numpy 分位数算法 × round/floor 穷举，无任何单一约定能复现原 20 格
（b=20 soft q25 = 0.9248 任何约定都得 0.92，原稿印 0.93）——坐实表格与
`m3_bootstrap_convergence_table.csv` 是一先一后两个版本的产物。
修法：按现行 CSV 以 **linear 分位数 + Decimal ROUND_HALF_UP 保留 2 位** 全表重算。
4 格变动：b=20 all-median 0.99→**1.00**、b=20 soft-q25 0.93→**0.92**、
b=30 all-q25 0.92→**0.91**、b=50 soft-q25 0.99→**1.00**；其余 16 格不变
（⭐两个 0.915 系格子精确值各落一边：all 0.914994→0.91 / soft 0.915023→0.92）。
multiset 受控差异 {0.99:−2, 1.00:+2, 0.93:−1, 0.91:+1} 与预期一致。
表注 "median ≈ 1.0 by b ≈ 20" 与新值更贴合（b=20 all-median 现印 1.00），无连带修改。

## Round 2（2026-09-01，wf_f3b87fc9-e26：6 lens[3回归+3新角度] + 2 对抗验证者；16 findings→14 CONFIRMED/1 MODIFIED/2 REJECTED-留强杀弱）

**收敛信号**：reg-figures（图引语义，逐 PDF 抽验）与 frontmatter（Highlights 字符数逐条实测 83/81/77/81/78、关键词 6 个、四声明块、全部图表被引）**零 finding 白卷**；proofread2 自评"处理完本批已接近零错收敛"。第一轮 60+ 处改动的回归审查未发现任何低级残骸。

**已应用（低级 7 处，multiset IDENTICAL）**：weight 重复删除（用验证者改判的带连字符版）/ a third to **a** half / L77·L469 逗号 / ±10% μ 括号归一 / OneDZ/Puetz 空格斜杠×2。

**已应用（judgment 8 处，用户批准，multiset 受控差异 6 项与预期一致）**：
- 甲1 §4.4 "only two…materially"→"two…most"（第一轮口径改写的最后一个漏网落点）
- 乙 L330 括注 "≈ 0.97–1.0 elsewhere" **CSV 证伪**（05FW008=0.67、SL0929=0.88 均在 elsewhere）→ "a median of ≈ 0.98 elsewhere"（其余 13 样本 frac_boot 中位数 0.982，LN+验证者双核）
- 甲2 §5.3 标题 trustworthy→robust（避免与下句撞词，呼应 §4.2 robustness check）
- 甲3 Fig4 注补 05FW008 total-口径说明句（两 vs 三例外的分母差异讲明）
- 甲4 Table 1 注补分母 "of the between-fit variance"
- 甲5 §1 删 "irreducible"（与全文 reducible 二分法直接矛盾的残留）
- 甲6 makes 主谓 / 丙8 References submitted-b 去 "Computers & Geosciences" 改中性（B 已离开 C&G，原句已不实）

**cover letter 更新草案（09_cover_letter_bayes.md，备份 .prepolish0901_bak）待用户过目**：日期 09-01 / 修回说明句（行号+编辑性更正披露）/ hedge 与新口径镜像稿件 / companion 披露改中性（submitted-b 不再声称在 C&G）。

**Data availability docx 并段悬案结案**：7 月投出的 docx 经亲验确实并成一段流水；md 已拆 7 段（零 token）。

**REJECTED 记录**：L329 双胞胎中动数字版（删 0.97–1.0 插 fifteen）被拦，幸存版+单独批准的乙条合并覆盖同一问题。

## 重建链验证（2026-09-02，第三轮终校进行中并行完成）

- **recipe 反推坐实**：用基线 md 做默认 pandoc 盲重建，`document.xml` 与 7 月投稿版**逐字节相同**（16 部件仅 core.xml 时间戳异）；Highlights 与 cover letter 的 docx 同样纯 pandoc 逐字节复现。→ 7 月三件 EM 条目 = `pandoc <md> --resource-path . -o <docx>`，无模板无预处理。
- **行号补丁**：采用 A 线 `build_r1_docx.py` 的 python-docx 插法（`lnNumType` 紧跟 `pgMar`，OOXML schema 顺序；08-11 已在 EM 成功 Build PDF）。⚠ 弃用 07-19 那份 `manuscript_bayes_uq_v2_linenumbers.docx` 的"追加在 sectPr 末尾"——不合 schema 顺序且从未被 Word/EM 打开过。字体/边距/行距**一律不动**，与已投版观感一致。
- **构建器**：`build_e_resubmission.py`（本目录），一次产出 `manuscript_bayes_uq_v3.docx`（带行号）+ `highlights_bayes_v3.docx` + `cover_letter_bayes_v3.docx`；内置对账：md 每个非空句段前 60 字符（归一化智能标点/XML 实体/列表标记后）必须出现在 docx 段落文本中。
- **干跑结果**（scratchpad）：稿件 584/584 探针命中、7274 词、lnNumType=1、Data availability 7/7 独立段；cover letter 52/52；Word COM 回读：正常打开、LineNumbering Active、CountBy 1、RestartContinuous。
- 正式产物待第三轮收官后 `python build_e_resubmission.py` 出到 `submission_package/`（v3 命名，7 月原件保留）。

## Round 3 · 终校（2026-09-02，wf_a889280d-b50：5 lens + 1 验证者；17 findings→11 CONFIRMED(1 MODIFIED)/6 REJECTED-去重或风格）

**收敛证明**：两名独立校对（正向逐行 / 逆向分块）在格式层面**均交白卷**（稿件仅 1 处算符空格）；第二轮回归零残骸；跨文件（稿件/Highlights/cover letter/六图/GA）零不一致。
**终校真正的收获在事实层（三方独立 CSV 核）**，用户全收：
- §4.1 "most **arc-poor** Songliao samples" → **craton-poor**（三个极端各向异性样本 w_S1 0.46–0.69、w_S3 ≈ 0.012；句尾自述 "craton weight fixed near zero"）
- §4.1 "within ~2° … for 11 of the 15" → "lies close to the S1–S2 direction for 11 of the 15 samples (within ~2° for most of them)"——`m3_eigen_results.csv` 实测 ≤2° 仅 8/15（第 9–11 名 6.5°/11.05°/15.4°，第 12 名跳到 60°），token 2/11/15 原样保留
- §5.5 "**The** Songliao samples … negligible cratonic component" → "**Most**"（SL0931 0.443 / S12653 0.591 / SL0932 0.872）
- 稿件 low：`≥3.10`→`≥ 3.10`。REJECTED：NMF/KL 缩写展开（稿件一贯不展开 MCMC/MAE/IQR 等标准缩写，单挑两个反成新不一致）、GA 文件名旧号（非文本错误，不重传）。
**cover letter**（7 条）：low 4 处（shows→reveals、Sub-sampling→Subsampling、**openness→openly**（7 月投出版即带的滑笔）、标题引号标点英式外置）；judgment 收 2 处（披露句改为具体点名 Table 2 重算 + prior-sensitivity 摘要重导、submit→resubmit）；**F 挂起**："the requested continuous line numbering" 把推测写成事实——07-15 退回信至今无人读过，须先读信再定稿（见下）。
**正式构建**：`build_e_resubmission.py` → `submission_package/*_v3.docx`；稿件 583/583 探针命中、lnNumType=1、Word COM 终检：行号 Active、7288 词、7 个内容哨兵全命中。cover letter v3 为临时版（待 F）。
**⚠ 预审规则（来自 A 线 06-02 退回信原文）**：Research Article **5,500 词上限 +10% 容差**，只计主文件、**排除 Abstract/Keywords/Highlights/References/Captions/Appendices**；须有名为 "Computer Code Availability" 的节；"You have one opportunity to correct all issues … failure … will result in desk rejection"。E 按此口径主体正文 4,148 词（Intro→Conclusions），全部计入含声明/表也仅 ~5.4k，**在限内**；但 Word 总计数 7,288 若被机械套用会越线——**以 07-15 信为准**。

## F 结案（2026-09-02）：07-15 退回信原件已读

原件 = `EM_SentBack_2026-07-15_CAGEO_E_line_numbers.eml`（本目录，自 Downloads 归档）。Date: Wed, 15 Jul 2026 14:21:32 +0000；Article Title 与 E 稿逐字一致。
**退回原因仅一条："- Line numbers"**（"Include line numbers in the document to facilitate the review process"）。无 Paper length、无 Code Availability。
→ cover letter "the requested continuous line numbering" 属实，原句保留；**投稿包三件 v3 docx + 六图全部终态，可上传**。
EM 事实：Incomplete 状态的 Action Links 只有 Edit/Remove，作者端无通信记录入口，退回信只存在于邮箱。

## 2026-09-02 · 上传前预检（续工 session，12:06–12:15）

- **只读交叉核全过**：三件 v3 docx 与 md 同分钟产出（11:32，无 stale）；`lnNumType` 恰 1 处（sectPr 内位于 footnotePr 之后，OOXML 顺序合法）；docx 图注 1→6、正文首引 1→6 严格递增（GA 描述的 "Fig. 3" 属前置件，7 月投出版同构已过预审）；Highlights 五条 83/81/77/81/78 字符 ≤ 85；六图 PDF 文本层零图号残留，Fig5 "peak width, Myr"、Fig6 "(companion study)/(this study)" 在位；重编号字节对账：旧 Fig2/3/6 = 新 Fig3/4/2 逐字节同，Fig5/6 为重渲染版。
- **改动 1 处**：cover letter 日期 1 → **2 September 2026**（`09_cover_letter_bayes.md` L7，备份 `09_cover_letter_bayes.pre0902date_bak.md`）；**只重建** `cover_letter_bayes_v3.docx`（12:14，同 recipe 纯 pandoc；document.xml 与前版仅该行差异，其余部件逐字节同）。稿件 / Highlights v3 docx 未动。⚠ 若不是 09-02 上传，改 L7 后 `pandoc 09_cover_letter_bayes.md --resource-path . -o submission_package/cover_letter_bayes_v3.docx` 即可。
- **⚠ 事实核出（待用户拍板，未改稿）**：B 线 **08-07 已遭 Minerals 桌面拒**（原件 Downloads `[Minerals] Manuscript ID_ minerals-4514253 - Declined for Publication.eml`，08-07 15:12），此后未再投 → E 稿三处当前不成立：§5.4 "(Pan, submitted-b; …)"、References "Pan, L., submitted-b … Manuscript submitted for publication."、cover letter "an analytic identifiability-bounds paper submitted for publication elsewhere"。选项：(a) 先投 B（SedGeo 包 08-01 已就绪）再 Approve E，一字不改；(b) 不改直接 Approve；(c) 三处改 "in preparation / unpublished manuscript" 后重建三件 docx（数字 multiset 不变；`build_e_resubmission.py` 哨兵 "Manuscript submitted for" 需同步改）。MASTER §1-B 的"08-07 疑待核"与 B RESUME 顶部"被动等"均已过时，未代改。

### (c) 已执行（2026-09-02 12:20，用户拍板"三处换 in preparation 重建"）

- 三处替换（备份 `08_manuscript_full.pre0902prep_bak.md` / `09_cover_letter_bayes.pre0902prep_bak.md`）：§5.4 "(Pan, submitted-b; …)" → "(Pan, in preparation; …)"；References "Pan, L., submitted-b. … Manuscript submitted for publication." → "Pan, L., in preparation. … Manuscript in preparation."；cover letter "paper submitted for publication elsewhere" → "paper in preparation"。`build_e_resubmission.py` 哨兵同步 "Manuscript submitted for" → "Manuscript in preparation"。
- 数字 multiset：稿件 967 / cover letter 21 token 均 IDENTICAL。`python build_e_resubmission.py` 重建三件 v3（12:20）：583/583 探针、lnNumType=1、Data availability 7/7、哨兵全中。
- 双路复核：① 与重建前 docx 逐段 diff——稿件恰 2 段（§5.4 + References 条目）、Highlights document.xml 逐字节同、cover letter 恰 1 段；② Word COM 实开：LineNumbering Active / CountBy 1 / 连续，Word 计数 7,289 词，"in preparation" 稿件 ×3、cover letter ×1，"submitted-b"/"submitted for publication" 均 0，日期 2 September 2026。
- 未动：`(Pan, submitted-a)` 6 处保留 "-a" 后缀（现已无 "-b" 配对，后缀无害；若要去后缀是 6 处 + References 1 条，用户拍板）。**投稿包三件 v3 + 六图 = 终态，可上传。**

### 追加：submitted-a 去后缀（2026-09-02 14:46 重建，用户拍板）

- `08_manuscript_full.md` 6 处 "submitted-a" → "submitted"（Intro §1 / §3.2 / §5.4 / References 条目 / Fig 1 注 / Fig 6 注；备份 `08_manuscript_full.pre0902suffix_bak.md`）。cover letter / Highlights / 构建脚本无此 token，未动。数字 multiset 967 IDENTICAL。
- 重建三件 v3：583/583 探针、lnNumType=1、7/7、哨兵全中。逐段 diff vs 12:20 版：稿件恰 6 段、Highlights 与 cover letter document.xml 逐字节同。Word COM 实开：行号 Active、"submitted-a" 0、"(Pan, submitted)" 4 + "(submitted)" 1 + "Pan, L., submitted." 1 = 6。
- companion 引用终态：A = (Pan, submitted)，B = (Pan, in preparation)。**投稿包三件 v3（本次时间戳）+ 六图 = 终态。**

### 09-02 16:40–17:10 参考文献扩充提案（未改稿，待拍板）

- 用户问"参考文献校对了？Zenodo 要改吗？文献太少"→ 17-agent workflow + 独立 Crossref 复核 44/44，产出 `REF_EXPANSION_PROPOSAL_0902.md`。发现摘要与待拍板项见 RESUME 续工入口 17:10 条（Liu 2017 两处错 / 期号 / B 线 Zenodo 应引 concept / Data availability 数据库 DOI 贴错 / 🔴 K1 层位疑点 / E 线 Zenodo v1.0.1 建议）。**未动稿件。**

### 09-02 17:20–17:50 用户拍板"都按建议改"→ 执行（改稿 + 重建 + QA + v1.0.1）

- 编辑脚本 `apply_ref_expansion.py`（56 处精确替换）+ `apply_qa_fixes.py`（15 处），全部断言命中；构建脚本哨兵改为 norm() 比较并新增三哨兵。
- References 13→39；K1→Jurassic–Cretaceous；Table 1 加 Formation 列；四处表述修正；B 线 Zenodo 改 concept。数字：正文 multiset 差异全部可归因（引用年份/样本号/新层位句 143,180,142,164,36/K1 的 '1'），结果数字零改动（QA 数字不变性 lens 独立确认）。
- 复核：逐段 diff（289→329→331 段；变化段数与预期一致）+ Word COM 两次（行号 Active、K1=0、旧 DOI=0、8,628 词）+ 4 路 QA。
- 详见 RESUME 续工入口 17:50 条。仓库 v1.0.1：commit 8d5e4a5 → GitHub release → Zenodo 10.5281/zenodo.22251075。

### 09-03 09:15–09:37 表格三线补丁（用户在 Word 发现表无底线）

- 根因：pandoc 3.9 默认 Table 样式只有 firstRow 条件格式的表头下边线，表格本身无 tblBorders。修法：`build_e_resubmission.py` 新增 `patch_table_borders()`（每表显式 top/bottom 单线 sz 8，left/right/inside nil，插在 tblW 之后合 schema 顺序），构建循环对三件都调用。
- 验证：先 `--out-dir` 建到 scratch → XML 两表各 1 组 tblBorders + Word COM 读边框（top=1/bottom=1/headerRowBottom=1/innerH 无）→ 用户关 Word 后正式重建（09:37），三件 document.xml 与 scratch 版逐字节相同，文字与 17:39 版逐段相同。

### 09-03 09:50–10:10 Table 1 横版补丁（EM Build PDF 实测暴露）

- 现象：EM 生成的 PDF（CAGEO-S-26-01685.pdf，34 页）其余全对（标题/摘要/行号 1–758 连续/39 条参考文献/六图顺序），但 Table 1 按 pandoc 的 10 等宽列（各 792 twips）硬排，样本号拆成 'R3137 0- S1265 0'、表头拆成 'Sampl e'、跨 3 页；Word 里正常只是 Word 自动调宽，EM 转换器不调。
- 修法：`build_e_resubmission.py` 新增 `patch_table1_landscape()`：'Tables' 标题前与 Notes 段末各插段级 sectPr（复制正文 sectPr 保留连续行号）形成竖–横–竖三节；pandoc 的 sectPr 原本无 pgSz/pgMar，三节全部显式 Letter + 1 in 页边距；Table 1 gridCol/tcW 显式列宽 1.15/1.20/0.90/0.60/0.65/0.35/0.95×3/1.30 = 9.0 in，表内 9 pt。自检改为 lnNumType 每节 1 条。
- 验证：XML 三 sectPr（第 2 节 orient=landscape 15840×12240）、160 格 tcW、正文逐段不变；Word COM：3 节方向 0/1/0、行号三节均 Active、Table 1 起止同页（p18）、列宽如设、边框 1/1。三件 v3 重建 10:07。

### 09-03 10:10–10:20 图清晰度核查 + Fig 1 图例改字 + GA 改矢量

- 用户反映 EM PDF 里图不清楚。实测：Fig 1–6 在 EM PDF 中为矢量整页嵌入（Type0/TrueType 字体、无位图、页面尺寸=图源尺寸），300 dpi 局部渲染锐利；**唯一真糊的是 GA 页**——EM 把 2201×2060 PNG 压成 540×577 JPEG（等效 72 dpi）。修法：GA 改传矢量 PDF `GraphicalAbstract_Fig3.pdf`（= Fig3_bayes_uncertainty.pdf 同源）。
- 顺带发现 Fig 1 图内图例仍是 'K1 DZ samples (this study)' 与 'Wide-rift / back-arc'，与改稿不一致 → 改仓库脚本 `Fig1_sample_locations_v1.py`（'DZ samples (this study)' / 'Songliao / Hailar / Erlian'，docstring 同步）重渲染，PNG 视觉验证通过，PDF 文字层无 K1/wide-rift，三格式已进 submission_package/figures 与仓库 figures_rendered（旧 Fig1 存 _superseded_0901/Fig1_sample_locations_0708.pdf）。Fig 2–6 与 GA 文字层扫描无残留。
- ⭐教训：EM 对 PDF 图源保留矢量，对 PNG/JPG 图源会下采样到 ~72 dpi——图一律传 PDF。仓库本地已 commit（未 push；v1.0.2 待用户点头）。

### 09-03 10:30–10:45 第二版 EM PDF 核查（CAGEO-S-26-01685 (1).pdf，33 页）

- 通过：Table 1 整表单个横版页（15 行完整）、GA 矢量（280 路径/无位图）、Fig 1 图例已更新、行号 1–749 严格连续、六图页面顺序正确、标题正确。
- 瑕疵：n 列 0.35 in 下 '111' 折成 '11/1' → 列宽改 0.45（Resolvability 1.30→1.20），重建 10:45，文字逐段不变，Word COM Table 1 仍单页。EM 需再换一次稿件 → Build → Approve。

### 09-03 10:55 第三版 EM PDF 核查 → 放行 Approve

- CAGEO-S-26-01685 (2).pdf，33 页：Table 1 单横版页、15 行完整、n=111 不再折行、顶/底线在；行号 1–749 严格连续；GA 与 Fig 1–6 矢量、顺序正确；标题/39 条文献正确。放行。
