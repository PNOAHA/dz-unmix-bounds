# -*- coding: utf-8 -*-
"""
E 线修回投稿包构建器（2026-09-01）
==================================
recipe 已经盲重建逐字节验证 = 7 月投稿版就是纯 pandoc、无模板、无预处理。
  1. pandoc 08_manuscript_full.md → manuscript_bayes_uq_v3.docx
  2. 打 lnNumType 行号补丁（python-docx，插在 pgMar 之后 = OOXML schema 顺序，
     A 线 R1 于 2026-08-11 在 EM 成功 Build PDF 的同款插法）
  3. pandoc highlights_bayes.md / 09_cover_letter_bayes.md → *_v3.docx
  4. 对账：md 句段抽样 ∈ docx 文本、Data availability 7 段独立、行号恰 1 处
跑法:  python build_e_resubmission.py [--out-dir DIR]   （默认 submission_package）
"""
import argparse, os, re, subprocess, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from docx import Document
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=os.path.join(HERE, "submission_package"))
OUT = ap.parse_args().out_dir; os.makedirs(OUT, exist_ok=True)

JOBS = [  # (源 md, 输出 docx, 是否打行号)
    (os.path.join(HERE, "08_manuscript_full.md"),                    "manuscript_bayes_uq_v3.docx",  True),
    (os.path.join(HERE, "submission_package", "highlights_bayes.md"), "highlights_bayes_v3.docx",     False),
    (os.path.join(HERE, "09_cover_letter_bayes.md"),                 "cover_letter_bayes_v3.docx",   False),
]

def pandoc(src, dst):
    r = subprocess.run(["pandoc", src, "--resource-path", HERE, "-o", dst], capture_output=True, text=True)
    if r.returncode: print("PANDOC FAIL:", r.stderr[:1500]); sys.exit(1)

def patch_linenumbers(path):
    doc = Document(path); n = 0
    for sec in doc.sections:
        sp = sec._sectPr
        if sp.find(qn("w:lnNumType")) is None:
            ln = sp.makeelement(qn("w:lnNumType"), {qn("w:countBy"): "1", qn("w:restart"): "continuous"})
            pg = sp.find(qn("w:pgMar")); (pg.addnext(ln) if pg is not None else sp.append(ln)); n += 1
    doc.save(path); return n

def patch_table_borders(path):
    """三线表：pandoc 3.9 默认 Table 样式只有表头下边线（firstRow 条件格式），表格本身无 tblBorders，
    于是没有顶线/底线（2026-09-03 用户在 Word 里发现）。显式给每个表加 top/bottom 单线，
    left/right/inside 置 nil；插在 tblW 之后 = CT_TblPr 的 schema 顺序（tblBorders 在 tblLayout 之前）。"""
    doc = Document(path); n = 0
    for tbl in doc.tables:
        pr = tbl._tbl.tblPr
        if pr.find(qn("w:tblBorders")) is not None: continue
        b = pr.makeelement(qn("w:tblBorders"), {})
        for edge, val, sz in [("top", "single", "8"), ("left", "nil", None), ("bottom", "single", "8"),
                              ("right", "nil", None), ("insideH", "nil", None), ("insideV", "nil", None)]:
            a = {qn("w:val"): val}
            if sz: a.update({qn("w:sz"): sz, qn("w:space"): "0", qn("w:color"): "auto"})
            b.append(b.makeelement(qn(f"w:{edge}"), a))
        w = pr.find(qn("w:tblW")); (w.addnext(b) if w is not None else pr.append(b)); n += 1
    doc.save(path); return n

import copy
def patch_table1_landscape(path):
    """Table 1（10 列）在 EM 的 PDF 转换器里按 pandoc 的 10 等宽列硬排，portrait 塞不下：
    样本号被拆成 'R3137 0- S1265 0'、跨 3 页（2026-09-03 EM Build PDF 实测）。Word 里正常只是因为 Word 自动调宽。
    修法：把 'Tables' 标题 + Table 1 + Notes 放进一个横版 section（前后各插 sectPr 段级分节符，复制正文 sectPr
    以保留 lnNumType 连续行号），并显式写 gridCol/tcW 列宽（合计 9.0 in = 横版可用宽度）、表内字号 9 pt。"""
    doc = Document(path)
    tbls = [t for t in doc.tables if len(t.columns) == 10]
    if len(tbls) != 1: print(f"   [landscape] 10 列表格 {len(tbls)} 个，跳过"); return 0
    tbl = tbls[0]; body = doc.element.body; kids = list(body)
    ti = kids.index(tbl._tbl)
    # 段落定位：表前最近的 'Tables' 标题；表后第一个以 'Notes:' 起头的段落
    def ptext(el): return "".join(x.text or "" for x in el.iter(qn("w:t")))
    head_i = next(i for i in range(ti - 1, -1, -1) if kids[i].tag == qn("w:p") and ptext(kids[i]).strip() == "Tables")
    notes_i = next(i for i in range(ti + 1, len(kids)) if kids[i].tag == qn("w:p") and ptext(kids[i]).lstrip().startswith("Notes:"))
    before_i = next(i for i in range(head_i - 1, -1, -1) if kids[i].tag == qn("w:p"))   # 'Tables' 前最后一个段落 → 结束 portrait 节
    main_sect = body.find(qn("w:sectPr"))
    def set_page(sp, landscape):
        """pandoc 的 sectPr 没有 pgSz/pgMar（靠 Word 默认，且随 locale 变 Letter/A4）——三节都显式写 Letter + 1 in 页边距，
        横版节交换宽高并标 orient；pgSz/pgMar 必须排在 lnNumType 之前（CT_SectPr schema 顺序）。"""
        for tag in ("w:pgSz", "w:pgMar"):
            e = sp.find(qn(tag))
            if e is not None: sp.remove(e)
        W, H = ("15840", "12240") if landscape else ("12240", "15840")
        a = {qn("w:w"): W, qn("w:h"): H}
        if landscape: a[qn("w:orient")] = "landscape"
        pgSz = sp.makeelement(qn("w:pgSz"), a)
        pgMar = sp.makeelement(qn("w:pgMar"), {qn("w:top"): "1440", qn("w:right"): "1440", qn("w:bottom"): "1440",
                                               qn("w:left"): "1440", qn("w:header"): "720", qn("w:footer"): "720", qn("w:gutter"): "0"})
        sp.insert(0, pgSz); sp.insert(1, pgMar)
    set_page(main_sect, landscape=False)          # 末节（图注）portrait，显式化
    def add_sect(p_el, landscape):
        sp = copy.deepcopy(main_sect); set_page(sp, landscape)
        ppr = p_el.find(qn("w:pPr"))
        if ppr is None: ppr = p_el.makeelement(qn("w:pPr"), {}); p_el.insert(0, ppr)
        ppr.append(sp)   # sectPr 是 pPr 的最后一个子元素（schema 顺序）
    add_sect(kids[before_i], landscape=False)
    add_sect(kids[notes_i], landscape=True)
    # 列宽（in）：Sample, Group, Formation, Lat, Lon, n, w_S1, w_S2, w_S3, Resolvability = 9.0 in
    widths = [1.15, 1.20, 0.90, 0.60, 0.65, 0.45, 0.95, 0.95, 0.95, 1.20]   # n 列 0.35→0.45（EM PDF 里 111 折行成 '11/1'，09-03 实测）
    tw = [int(round(w * 1440)) for w in widths]
    tblPr = tbl._tbl.tblPr
    tblW = tblPr.find(qn("w:tblW")); tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(sum(tw)))
    grid = tbl._tbl.find(qn("w:tblGrid"))
    for gc, w in zip(grid.findall(qn("w:gridCol")), tw): gc.set(qn("w:w"), str(w))
    for row in tbl.rows:
        for cell, w in zip(row.cells, tw):
            tcPr = cell._tc.get_or_add_tcPr(); tcW = tcPr.find(qn("w:tcW"))
            if tcW is None: tcW = tcPr.makeelement(qn("w:tcW"), {}); tcPr.insert(0, tcW)
            tcW.set(qn("w:type"), "dxa"); tcW.set(qn("w:w"), str(w))
            for p in cell.paragraphs:
                for r in p.runs: r.font.size = __import__("docx").shared.Pt(9)
    doc.save(path); return 1

import html
def norm(s):  # 两侧同用：抹平 pandoc 智能标点与 XML 实体，只比"内容"
    s = html.unescape(s)
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("…", "..."), (" ", " ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)
def docx_text(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    d = Document(path)   # 按段落/单元格取文本：run 边界不产生伪空格
    parts = [p.text for p in d.paragraphs] + [c.text for t in d.tables for r in t.rows for c in r.cells]
    return xml, norm(chr(10).join(parts))

problems = []
for src, name, ln in JOBS:
    dst = os.path.join(OUT, name); pandoc(src, dst)
    if ln: patch_linenumbers(dst)
    patch_table_borders(dst)
    if ln: patch_table1_landscape(dst)
    xml, text = docx_text(dst)
    md = open(src, encoding="utf-8").read()
    # 抽样对账：md 每个非空、非表格、非标题行取前 60 字符（去 markdown 记号）必须出现在 docx 文本里
    miss = 0; checked = 0
    for line in md.split("\n"):
        s = line.strip()
        if not s or s.startswith(("|", "#", "---", "![", "- ")) or "**" in s[:3] and s.endswith("**"): continue
        s = re.sub(r"^(\d+\.|-)\s+", "", s)          # 列表标记 → pandoc 自动编号，不在文本里
        probe = norm(re.sub(r"[*`]|\\\\", "", s))[:60].strip()   # 保留 _（w_S3 等标识符原样进 docx）
        if len(probe) < 25: continue
        checked += 1
        if probe not in text:
            miss += 1
            if miss <= 8:
                key = probe[10:32]; j = text.find(key)
                print("   MISS:", repr(probe)); print("        docx侧:", repr(text[max(0,j-70):j+70]) if j >= 0 else "(中段 22 字符也找不到)")
    lnc = xml.count("<w:lnNumType")
    words = len(text.split())
    print(f"{name}: {words} 词 | 抽样 {checked} 行缺 {miss} | lnNumType={lnc} / sectPr={xml.count(chr(60)+chr(119)+chr(58)+chr(115)+chr(101)+chr(99)+chr(116)+chr(80)+chr(114))}")
    if miss: problems.append(f"{name}: {miss} 行 md 句段未在 docx 找到")
    nsec = xml.count("<w:sectPr")
    if ln and (lnc < 1 or lnc != nsec): problems.append(f"{name}: lnNumType 应每节 1 条（{nsec} 节）实为 {lnc}")
    if "manuscript" in name:
        d = Document(dst); keys = ["Name of the code/library", "Developer and contact", "Year first available",
               "Program language", "Software required", "Program size", "Availability:"]
        seps = sum(1 for p in d.paragraphs if any(p.text.startswith(k) for k in keys))
        print(f"   Data availability 独立段: {seps}/7 | 段落总数 {len(d.paragraphs)}")
        if seps != 7: problems.append(f"Data availability 段落 {seps}/7")
        for must in ["median of under 3%", "Manuscript in preparation", "Jurassic–Cretaceous samples", "Puetz et al., 2024", "Li et al., 2026", "(Fig. 6)", "peak width", "robust and prior-insensitive"]:
            if norm(must) not in text: problems.append(f"哨兵缺失: {must}")
print("\n" + ("✅ 全部对账通过 → " + OUT if not problems else "🔴 问题:\n  " + "\n  ".join(problems)))
