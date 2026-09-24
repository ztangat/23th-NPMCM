# -*- coding: utf-8 -*-
"""把 make_paper.py（docx 版论文脚本）转译为 Markdown 版论文。

做法：用一套"假 python-docx"替换 docx 模块，执行 make_paper.py 的源码，
把段落/表格按出现顺序收集起来，再渲染为 Markdown。
不落任何 docx 文件。
"""
import os
import re
import sys
import types

SRC = r'C:\Users\dkyyt\Desktop\F题\HY\_tools\make_paper.py'
OUT = r'C:\Users\dkyyt\Desktop\F题\HY\参赛论文_算力约束下提升大语言模型能力的资源配置建模.md'

BODY = []


# ---------------- 假 docx ----------------
class PF(object):
    def __init__(self):
        self.first_line_indent = 0
        self.left_indent = 0
        self.space_before = 0
        self.space_after = 0
        self.line_spacing_rule = None


class Font(object):
    def __init__(self):
        self.size = None
        self.name = None
        self.bold = False
        self.italic = False


class RPr(object):
    def find(self, k):
        return None

    def append(self, x):
        pass


class RFonts(object):
    def set(self, k, v):
        pass


class Elem(object):
    def get_or_add_rPr(self):
        if not hasattr(self, '_rPr'):
            self._rPr = RPr()
        return self._rPr

    def addnext(self, x):
        pass


class Run(object):
    def __init__(self, text=''):
        self.text = text
        self.font = Font()
        self._element = Elem()
        self._r = Elem()


class Paragraph(object):
    def __init__(self):
        self.runs = []
        self.alignment = None
        self.paragraph_format = PF()

    def add_run(self, t=''):
        r = Run(t)
        self.runs.append(r)
        return r

    def clear(self):
        self.runs = []

    @property
    def text(self):
        return ''.join(r.text for r in self.runs)


class Cell(object):
    def __init__(self):
        self.text = ''
        self.paragraphs = [Paragraph()]


class Table(object):
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.style = None
        self.alignment = None
        self._cells = {}
        for i in range(rows):
            for j in range(cols):
                self._cells[(i, j)] = Cell()

    def cell(self, i, j):
        return self._cells[(i, j)]


class Header(object):
    def __init__(self):
        self.is_linked_to_previous = True
        self.paragraphs = [Paragraph()]

    def add_paragraph(self):
        p = Paragraph()
        self.paragraphs.append(p)
        return p


class Section(object):
    def __init__(self):
        self.page_width = self.page_height = 0
        self.left_margin = self.right_margin = 0
        self.top_margin = self.bottom_margin = 0
        self.header = Header()
        self.footer = Header()


class Style(object):
    def __init__(self):
        self.font = Font()
        self.paragraph_format = PF()
        self.element = types.SimpleNamespace(
            rPr=types.SimpleNamespace(rFonts=RFonts()))


class FakeDoc(object):
    def __init__(self):
        self.sections = [Section()]
        self.styles = {'Normal': Style()}

    def add_paragraph(self):
        p = Paragraph()
        BODY.append(('p', p))
        return p

    def add_table(self, rows=0, cols=0):
        t = Table(rows, cols)
        BODY.append(('t', t))
        return t

    def add_page_break(self):
        BODY.append(('brk', None))

    def save(self, path):
        pass


def _Pt(v):
    return float(v)


def _Cm(v):
    return float(v)


class _OxmlElement(object):
    def __init__(self, tag):
        self.tag = tag

    def set(self, k, v):
        pass


def _qn(s):
    return s


def install_fake_docx():
    docx = types.ModuleType('docx')
    docx.Document = lambda: FakeDoc()
    shared = types.ModuleType('docx.shared')
    shared.Pt, shared.Cm = _Pt, _Cm
    shared.RGBColor = lambda *a, **k: None
    etext = types.ModuleType('docx.enum.text')
    etext.WD_ALIGN_PARAGRAPH = types.SimpleNamespace(CENTER=1, LEFT=0, RIGHT=2)
    etext.WD_LINE_SPACING = types.SimpleNamespace(SINGLE=0)
    etable = types.ModuleType('docx.enum.table')
    etable.WD_TABLE_ALIGNMENT = types.SimpleNamespace(CENTER=1)
    esec = types.ModuleType('docx.enum.section')
    esec.WD_SECTION = types.SimpleNamespace(NEW_PAGE=2)
    oxns = types.ModuleType('docx.oxml.ns')
    oxns.qn = _qn
    oxml = types.ModuleType('docx.oxml')
    oxml.OxmlElement = _OxmlElement
    oxml.ns = oxns
    enum = types.ModuleType('docx.enum')
    enum.text = etext
    enum.table = etable
    enum.section = esec
    docx.shared = shared
    docx.enum = enum
    docx.oxml = oxml
    for name, mod in [('docx', docx), ('docx.shared', shared),
                      ('docx.enum', enum), ('docx.enum.text', etext),
                      ('docx.enum.table', etable), ('docx.enum.section', esec),
                      ('docx.oxml', oxml), ('docx.oxml.ns', oxns)]:
        sys.modules[name] = mod


install_fake_docx()
with open(SRC, 'r', encoding='utf-8') as f:
    code = f.read()
g = {'__name__': '__main__', '__file__': SRC}
exec(compile(code, SRC, 'exec'), g)


# ---------------- 渲染 ----------------
def esc(s):
    return s.replace('|', '\\|')


def render_paragraph(p):
    runs = [r for r in p.runs if r.text != '']
    if not runs:
        return None
    txt = ''.join(r.text for r in runs)
    r0 = runs[0]
    size = r0.font.size
    name = r0.font.name
    bold = any(r.font.bold for r in runs)
    indent = p.paragraph_format.first_line_indent or 0
    center = (p.alignment == 1)

    # 公式
    if name == 'Cambria Math':
        main = ''.join(r.text for r in runs if r.font.name == 'Cambria Math')
        lab = ''.join(r.text for r in runs if r.font.name != 'Cambria Math').strip()
        if lab:
            return '$$ %s \\qquad %s $$' % (main.strip(), lab)
        return '$$ %s $$' % main.strip()

    if size == 16.0:
        return '# %s' % txt.strip()
    if size == 14.0:
        return '## %s' % txt.strip()
    if bold and indent == 0 and not center:
        return '### %s' % txt.strip()
    if center and not bold:
        return '<p align="center">%s</p>' % txt.strip()
    if bold:
        return '> **%s**' % txt.strip()
    return txt.strip()


def render_table(t):
    rows = []
    for i in range(t.rows):
        row = []
        for j in range(t.cols):
            c = t.cell(i, j)
            v = c.text if c.text else ''.join(r.text for r in c.paragraphs[0].runs)
            row.append(str(v).strip())
        rows.append(row)
    out = []
    out.append('| ' + ' | '.join(esc(x) for x in rows[0]) + ' |')
    out.append('|' + '|'.join([':---:'] * len(rows[0])) + '|')
    for row in rows[1:]:
        out.append('| ' + ' | '.join(esc(x) for x in row) + ' |')
    return '\n'.join(out)


lines = []
for kind, obj in BODY:
    if kind == 'brk':
        continue
    if kind == 'p':
        s = render_paragraph(obj)
        if s is None:
            continue
        lines.append(s)
        lines.append('')
    else:
        lines.append(render_table(obj))
        lines.append('')

md = '\n'.join(lines)
md = re.sub(r'\n{3,}', '\n\n', md)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(md)
print('saved:', OUT, len(md), 'chars')
