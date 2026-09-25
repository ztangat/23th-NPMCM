# -*- coding: utf-8 -*-
"""
论文 md -> docx 转换脚本（严格对齐《华为杯第二十三届中国研究生数学建模竞赛论文格式规范》）
规范要点：
  - 论文题目：三号黑体，居中
  - 一级标题：四号黑体，居中
  - 其他汉字：小四号宋体，单倍行距
  - 页码：页脚居中，阿拉伯数字，从摘要页起连续编号
  - 无页眉，无身份信息
说明：本程序及代码是在人工智能工具辅助下完成的。
      工具名称：DeepSeek-V3（deepseek-chat），开发机构：深度求索（DeepSeek）。
"""
import os, re, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # q_all/
MD   = os.path.join(ROOT, "reports", "论文_算力约束下提升大语言模型能力的资源配置建模.md")
OUT  = os.path.join(ROOT, "reports", "论文_算力约束下提升大语言模型能力的资源配置建模.docx")

# ---------- 字号 ----------
SZ_TITLE   = Pt(16)   # 三号 ≈ 16pt
SZ_H1      = Pt(14)   # 四号 ≈ 14pt
SZ_H2      = Pt(12)   # 小四
SZ_BODY    = Pt(12)   # 小四 ≈ 12pt
SZ_TABLE   = Pt(10.5) # 五号
SZ_CAPTION = Pt(10.5)

HEI = "SimHei"     # 黑体
SONG = "SimSun"    # 宋体
KAI  = "KaiTi"     # 楷体（公式/代码用等宽替代）


def set_run(run, font=SONG, size=SZ_BODY, bold=False, italic=False, color=None):
    run.font.name = font
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), font)
    rFonts.set(qn('w:hAnsi'), font)
    rFonts.set(qn('w:eastAsia'), font)
    rFonts.set(qn('w:cs'), font)


def set_para_format(p, align=None, line_single=True, before=0, after=0,
                    first_line_indent=None, left_indent=None, keep_with_next=False):
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if line_single:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    if left_indent is not None:
        pf.left_indent = left_indent
    pf.keep_with_next = keep_with_next


# ---------- 页脚页码 ----------
def add_page_number(footer_para):
    """在页脚段落插入 PAGE 域，居中，阿拉伯数字"""
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_para.add_run()
    set_run(run, font=SONG, size=Pt(10.5))
    fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText'); instrText.set(qn('xml:space'), 'preserve'); instrText.text = 'PAGE'
    fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
    run._element.append(fldChar1)
    run._element.append(instrText)
    run._element.append(fldChar2)


# ---------- 行内格式解析 ----------
def add_inline(p, text, base_font=SONG, base_size=SZ_BODY, base_bold=False):
    """解析 **粗体**、$公式$、`代码`，其余普通"""
    # 用 正则 token 化：**...** | $...$ | `...`
    pattern = re.compile(r'(\*\*.+?\*\*|\$[^$]+\$|`[^`]+`)', re.S)
    parts = pattern.split(text)
    for seg in parts:
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**') and len(seg) > 4:
            r = p.add_run(seg[2:-2]); set_run(r, base_font, base_size, bold=True)
        elif seg.startswith('$') and seg.endswith('$') and len(seg) > 2:
            # 行内公式：转字面（去 LaTeX 标记的简易渲染），用斜体宋体
            r = p.add_run(_latex_to_text(seg[1:-1]))
            set_run(r, base_font, base_size, italic=True)
        elif seg.startswith('`') and seg.endswith('`') and len(seg) > 2:
            r = p.add_run(seg[1:-1]); set_run(r, "Consolas", base_size - Pt(1))
        else:
            r = p.add_run(seg); set_run(r, base_font, base_size, bold=base_bold)


def _latex_to_text(s):
    """把常见 LaTeX 片段转成可读 Unicode，尽量保留可读性"""
    s = s.strip()
    repl = [
        (r'\\times', '×'), (r'\\cdot', '·'), (r'\\approx', '≈'), (r'\\le', '≤'),
        (r'\\ge', '≥'), (r'\\neq', '≠'), (r'\\to', '→'), (r'\\quad', ' '),
        (r'\\alpha', 'α'), (r'\\beta', 'β'), (r'\\gamma', 'γ'), (r'\\eta', 'η'),
        (r'\\rho', 'ρ'), (r'\\tau', 'τ'), (r'\\theta', 'θ'), (r'\\sigma', 'σ'),
        (r'\\Delta', 'Δ'), (r'\\log_', 'log'), (r'\\ln', 'ln'), (r'\\max', 'max'),
        (r'\\min', 'min'), (r'\\sum', '∑'), (r'\\infty', '∞'),
        (r'\\partial', '∂'), (r'\\sqrt', '√'),
        (r'\\left', ''), (r'\\right', ''), (r'\\text\{([^}]*)\}', r'\1'),
        (r'\\boxed\{([^{}]*)\}', r'\1'),
        (r'\{', ''), (r'\}', ''), (r'\\', ''),
        (r'\^\{([^}]*)\}', r'^(\1)'), (r'_\{([^}]*)\}', r'_\1'),
        (r'\^(\w)', r'^\1'), (r'_(\w)', r'_\1'),
    ]
    for a, b in repl:
        s = re.sub(a, b, s)
    s = s.replace('  ', ' ').strip()
    return s


def add_display_formula(doc, latex):
    """居中显示的公式块（用等宽字，简化渲染）"""
    p = doc.add_paragraph()
    set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=4)
    r = p.add_run(_latex_to_text(latex))
    set_run(r, "Cambria Math", SZ_BODY)
    r.font.italic = False
    return p


def add_code_block(doc, code):
    p = doc.add_paragraph()
    set_para_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=2, after=2,
                    left_indent=Cm(0.6))
    r = p.add_run(code)
    set_run(r, "Consolas", Pt(9))
    # 底纹
    pPr = p._element.get_or_add_pPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)


def add_table(doc, rows):
    """rows: list[list[str]]，第一行为表头"""
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = 'Table Grid'
    t.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ''
            p = cell.paragraphs[0]
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=1, after=1)
            txt = row[j] if j < len(row) else ''
            bold = (i == 0)
            add_inline(p, txt, base_font=SONG, base_size=SZ_TABLE, base_bold=bold)
            # 表头底纹
            if i == 0:
                tcPr = cell._element.get_or_add_tcPr()
                shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), 'D9E2F3')
                tcPr.append(shd)
    # 表后空行
    sp = doc.add_paragraph(); set_para_format(sp, after=2)
    return t


def add_image(doc, img_path):
    if not os.path.exists(img_path):
        print("  [警告] 图片缺失:", img_path); return
    with Image.open(img_path) as im:
        w, h = im.size
    page_w_cm = 15.5   # 可用宽度
    max_h_cm = 11.0
    ratio = h / w if w else 1
    disp_w = page_w_cm
    disp_h = disp_w * ratio
    if disp_h > max_h_cm:
        disp_h = max_h_cm
        disp_w = disp_h / ratio
    p = doc.add_paragraph()
    set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=2)
    r = p.add_run()
    r.add_picture(img_path, width=Cm(disp_w), height=Cm(disp_h))


# ---------- 解析 md ----------
def parse_md(md_text):
    lines = md_text.split('\n')
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        # 表格
        if s.startswith('|') and i + 1 < n and re.match(r'^\|[\s:\-|]+\|$', lines[i+1].strip()):
            tbl = [s]
            i += 2
            while i < n and lines[i].strip().startswith('|'):
                tbl.append(lines[i].strip()); i += 1
            rows = []
            for tl in tbl:
                cells = [c.strip() for c in tl.strip('|').split('|')]
                rows.append(cells)
            blocks.append(('table', rows))
            continue
        # 代码块
        if s.startswith('```'):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(lines[i]); i += 1
            i += 1
            blocks.append(('code', '\n'.join(buf)))
            continue
        # 公式块 $$...$$
        if s.startswith('$$'):
            buf = [s[2:]]
            if s.endswith('$$') and len(s) > 4:
                blocks.append(('formula', s[2:-2])); i += 1; continue
            i += 1
            while i < n and not lines[i].strip().endswith('$$'):
                buf.append(lines[i]); i += 1
            if i < n:
                buf.append(lines[i].strip()[:-2]); i += 1
            blocks.append(('formula', ' '.join(buf).strip()))
            continue
        # 标题
        m = re.match(r'^(#{1,4})\s+(.*)$', s)
        if m:
            blocks.append(('h%d' % len(m.group(1)), m.group(2).strip())); i += 1; continue
        # 图片
        m = re.match(r'^!\[.*?\]\((.*?)\)$', s)
        if m:
            blocks.append(('img', m.group(1))); i += 1; continue
        # 水平线
        if s in ('---', '***', '___'):
            blocks.append(('hr', '')); i += 1; continue
        # 引用
        if s.startswith('*') and s.endswith('*') and len(s) > 2 and not s.startswith('**'):
            blocks.append(('caption', s.strip('*'))); i += 1; continue
        # 列表
        m = re.match(r'^(?:[-*]|\d+\.)\s+(.*)$', s)
        if m:
            blocks.append(('li', m.group(1))); i += 1; continue
        # 空行
        if s == '':
            blocks.append(('blank', '')); i += 1; continue
        # 普通段落
        blocks.append(('p', s)); i += 1
    return blocks


def build():
    with open(MD, 'r', encoding='utf-8') as f:
        md = f.read()
    blocks = parse_md(md)

    doc = Document()
    # 页面设置：A4，页边距
    sec = doc.sections[0]
    sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.8); sec.right_margin = Cm(2.7)
    sec.header_distance = Cm(1.5); sec.footer_distance = Cm(1.75)
    sec.different_first_page_header_footer = False

    # 默认样式
    style = doc.styles['Normal']
    style.font.name = SONG; style.font.size = SZ_BODY
    style.element.rPr.rFonts.set(qn('w:eastAsia'), SONG)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    style.paragraph_format.space_after = Pt(0)

    # 页脚页码
    add_page_number(sec.footer.paragraphs[0])

    IMG_BASE_REL = os.path.join(ROOT, "figures")  # md 中 figures/xx 相对 reports/，但图在 q_all/figures

    first_h1_done = False
    for kind, val in blocks:
        if kind == 'h1':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=10)
            add_inline(p, val, base_font=HEI, base_size=SZ_TITLE, base_bold=False)
        elif kind == 'h2':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=10, after=6, keep_with_next=True)
            add_inline(p, val, base_font=HEI, base_size=SZ_H1, base_bold=False)
        elif kind == 'h3':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4, keep_with_next=True)
            add_inline(p, val, base_font=HEI, base_size=SZ_H2, base_bold=True)
        elif kind == 'h4':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=3, keep_with_next=True)
            add_inline(p, val, base_font=HEI, base_size=SZ_BODY, base_bold=True)
        elif kind == 'p':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=3,
                            first_line_indent=Pt(24))
            add_inline(p, val)
        elif kind == 'li':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=2,
                            left_indent=Cm(0.75), first_line_indent=Pt(-12))
            r = p.add_run('● '); set_run(r, SONG, SZ_BODY)
            add_inline(p, val)
        elif kind == 'formula':
            add_display_formula(doc, val)
        elif kind == 'code':
            add_code_block(doc, val)
        elif kind == 'table':
            add_table(doc, val)
        elif kind == 'img':
            rel = val.replace('\\', '/')
            fn = os.path.basename(rel)
            path = os.path.join(IMG_BASE_REL, fn)
            add_image(doc, path)
        elif kind == 'caption':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6)
            add_inline(p, val, base_font=SONG, base_size=SZ_CAPTION)
        elif kind == 'hr':
            p = doc.add_paragraph()
            set_para_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=2)
            r = p.add_run('—' * 20); set_run(r, SONG, Pt(8), color=RGBColor(0x99, 0x99, 0x99))
        elif kind == 'blank':
            continue

    doc.save(OUT)
    print("[OK] 已生成:", OUT)
    print("     段落数:", len(doc.paragraphs), " 表格数:", len(doc.tables))


if __name__ == '__main__':
    build()
