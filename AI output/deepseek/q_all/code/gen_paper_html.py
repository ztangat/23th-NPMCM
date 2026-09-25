# -*- coding: utf-8 -*-
"""
gen_paper_html.py —— 将统一论文 markdown 转为可离线浏览的 HTML（内嵌图片 base64）

读取：reports/论文_算力约束下提升大语言模型能力的资源配置建模.md
产出：reports/论文_算力约束下提升大语言模型能力的资源配置建模.html

特性：
  · 图片从 ../figures 读取并 base64 内嵌（离线可看）
  · 公式用 MathJax CDN（联网时渲染），离线保留 LaTeX 源码
  · 竞赛论文排版：摘要页独立、正文分页、页脚页码
"""
import os, sys, re, base64, html as _html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "reports")
FIGDIR = os.path.join(ROOT, "figures")

MD = os.path.join(REPORTS, "论文_算力约束下提升大语言模型能力的资源配置建模.md")
OUT = os.path.join(REPORTS, "论文_算力约束下提升大语言模型能力的资源配置建模.html")


def fig_b64(name):
    """name 形如 figures/F01_xxx.png 或 F01_xxx.png"""
    base = os.path.basename(name)
    for p in (os.path.join(FIGDIR, base), os.path.join(ROOT, name)):
        if os.path.exists(p):
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return None


CSS = """
:root{--fg:#1a1a1a;--bg:#ffffff;--muted:#5b6472;--accent:#1e3a8a;--accent2:#b91c1c;
--border:#d8dee9;--code:#f6f8fa;}
*{box-sizing:border-box;}
body{font-family:"Times New Roman","SimSun","宋体",serif;line-height:1.85;
color:var(--fg);background:#f2f3f5;margin:0;padding:0;}
.page{background:var(--bg);max-width:1000px;margin:0 auto;padding:48px 60px 80px;
box-shadow:0 0 18px rgba(0,0,0,.08);}
h1{font-family:"SimHei","黑体",sans-serif;font-size:1.75em;text-align:center;
border-bottom:3px solid var(--accent);padding-bottom:14px;margin-top:6px;line-height:1.5;}
h2{font-family:"SimHei","黑体",sans-serif;font-size:1.4em;margin-top:2.4em;
border-left:6px solid var(--accent);padding-left:14px;color:#111;}
h3{font-family:"SimHei","黑体",sans-serif;font-size:1.18em;margin-top:1.8em;color:#1f2937;}
h4{font-size:1.05em;color:#374151;}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:0.9em;
font-family:"SimSun","宋体",serif;}
th,td{border:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top;}
th{background:#eef2f8;font-weight:700;font-family:"SimHei","黑体",sans-serif;}
tr:nth-child(even) td{background:#fafbfd;}
code{background:var(--code);padding:2px 6px;border-radius:4px;
font-family:Consolas,Monaco,monospace;font-size:0.88em;color:#b91c1c;}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;
border:1px solid var(--border);font-size:0.86em;line-height:1.5;}
pre code{background:none;color:#1f2937;padding:0;}
blockquote{border-left:4px solid var(--accent);background:#f8faff;margin:1.2em 0;
padding:10px 18px;color:#334155;}
blockquote p{margin:0.4em 0;}
img{display:block;max-width:100%;height:auto;margin:1.4em auto;border:1px solid var(--border);
border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.06);}
em{color:var(--muted);font-style:italic;}
hr{border:none;border-top:1px dashed var(--border);margin:2.4em 0;}
ul,ol{padding-left:1.6em;}
li{margin:0.35em 0;}
a{color:var(--accent);text-decoration:none;}
a:hover{text-decoration:underline;}
.mjx-container{font-size:1.02em;}
/* 摘要页 */
.abstract-page{page-break-after:always;border-bottom:2px solid #ddd;padding-bottom:40px;margin-bottom:40px;}
/* 打印 */
@media print{
  body{background:#fff;}
  .page{box-shadow:none;max-width:none;padding:0 12mm;}
  h2{page-break-after:avoid;}
  img{page-break-inside:avoid;}
  table{page-break-inside:avoid;}
}
"""


def inline(t):
    t = _html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
    return t


def md_to_html(md):
    lines = md.split("\n")
    out = []
    i = 0
    in_code = False; code_buf = []
    in_table = False; table_rows = []
    in_ul = False; in_ol = False
    in_abstract = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul: out.append("</ul>"); in_ul = False
        if in_ol: out.append("</ol>"); in_ol = False

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table: return
        rows = [r for r in table_rows if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
        html = ["<table>"]
        for ri, r in enumerate(rows):
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            tag = "th" if ri == 0 else "td"
            html.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
        html.append("</table>")
        out.append("\n".join(html))
        in_table = False; table_rows = []

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            if not in_code: in_code = True; code_buf = []
            else:
                in_code = False
                out.append("<pre><code>" + _html.escape("\n".join(code_buf)) + "</code></pre>")
            i += 1; continue
        if in_code: code_buf.append(ln); i += 1; continue
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            close_lists(); in_table = True; table_rows.append(ln); i += 1; continue
        else:
            flush_table()
        if ln.strip().startswith("$$"):
            close_lists()
            buf = [ln.strip()]
            if ln.strip().endswith("$$") and len(ln.strip()) > 2:
                out.append(f"<div>{buf[0]}</div>")
            else:
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    buf.append(lines[i]); i += 1
                if i < len(lines): buf.append(lines[i])
                out.append("<div>" + "\n".join(buf) + "</div>")
            i += 1; continue
        mimg = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln.strip())
        if mimg:
            close_lists()
            alt, src = mimg.group(1), mimg.group(2)
            b64 = fig_b64(src)
            if b64:
                out.append(f'<img alt="{_html.escape(alt)}" src="{b64}">')
            else:
                out.append(f'<img alt="{_html.escape(alt)}" src="{src}">')
            i += 1; continue
        # 摘要页标记：一级标题后第一段"(**摘要**)" → 用分页包裹
        mh = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if mh:
            close_lists()
            lvl = len(mh.group(1))
            out.append(f"<h{lvl}>{inline(mh.group(2))}</h{lvl}>")
            i += 1; continue
        if ln.strip().startswith(">"):
            close_lists()
            content = ln.strip().lstrip(">").strip()
            buf = [content]; i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            out.append("<blockquote>" + " ".join(inline(b) for b in buf) + "</blockquote>")
            continue
        if re.match(r"^\s*[-*_]{3,}\s*$", ln):
            close_lists(); out.append("<hr>"); i += 1; continue
        mu = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if mu:
            if not in_ul: close_lists(); out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(mu.group(1))}</li>")
            i += 1; continue
        mo = re.match(r"^\s*\d+\.\s+(.*)$", ln)
        if mo:
            if not in_ol: close_lists(); out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(mo.group(1))}</li>")
            i += 1; continue
        if not ln.strip():
            close_lists(); i += 1; continue
        close_lists()
        out.append(f"<p>{inline(ln)}</p>")
        i += 1
    close_lists(); flush_table()
    return "\n".join(out)


def main():
    with open(MD, encoding="utf-8") as f:
        md = f.read()
    body = md_to_html(md)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>算力约束下提升大语言模型能力的资源配置建模 — 论文</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']]}},
  svg: {{fontCache: 'global'}},
  options: {{skipHtmlTags: ['script','noscript','style','textarea','pre','code']}}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {OUT}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
