# -*- coding: utf-8 -*-
"""
gen_html_report.py —— 将问题二 markdown 报告转为可离线浏览的 HTML（内嵌图片 base64）

读取：reports/问题二_完整解答报告.md
产出：reports/问题二_完整解答报告.html
特性：图片 base64 内嵌（离线可看）；公式用 MathJax CDN（联网时渲染），
      若离线则保留 LaTeX 源码（仍可读）。
"""
import os, sys, re, base64, html as _html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MD = os.path.join(REPORTS, "问题二_完整解答报告.md")
OUT = os.path.join(REPORTS, "问题二_完整解答报告.html")


def fig_b64(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        # 尝试直接给的名字
        p2 = os.path.join(FIG, os.path.basename(name))
        if os.path.exists(p2):
            p = p2
        else:
            return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


CSS = """
:root{--fg:#1a1a1a;--bg:#ffffff;--muted:#5b6472;--accent:#2563eb;--border:#e3e8ef;--code:#f6f8fa;}
*{box-sizing:border-box;}
body{font-family:"Segoe UI","Microsoft YaHei",-apple-system,sans-serif;line-height:1.8;
color:var(--fg);background:var(--bg);max-width:1000px;margin:0 auto;padding:40px 28px 90px;}
h1{font-size:1.9em;border-bottom:3px solid var(--accent);padding-bottom:12px;margin-top:8px;line-height:1.4;}
h2{font-size:1.5em;margin-top:2.4em;border-left:5px solid var(--accent);padding-left:12px;}
h3{font-size:1.2em;margin-top:1.8em;color:#1f2937;}
h4{font-size:1.05em;color:#374151;}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:0.92em;}
th,td{border:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top;}
th{background:#f0f4fa;font-weight:600;}
tr:nth-child(even) td{background:#fafbfd;}
code{background:var(--code);padding:2px 6px;border-radius:4px;font-family:Consolas,Monaco,monospace;
font-size:0.9em;color:#b91c1c;}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;border:1px solid var(--border);}
pre code{background:none;color:#1f2937;padding:0;}
blockquote{border-left:4px solid var(--accent);background:#f8faff;margin:1.2em 0;
padding:10px 18px;color:#334155;border-radius:0 6px 6px 0;}
blockquote p{margin:0.4em 0;}
img{display:block;max-width:100%;height:auto;margin:1.4em auto;border:1px solid var(--border);
border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.06);}
em{color:var(--muted);}
hr{border:none;border-top:1px solid var(--border);margin:2.4em 0;}
ul,ol{padding-left:1.6em;}
li{margin:0.3em 0;}
a{color:var(--accent);}
.mjx-container{font-size:1.02em;}
"""


def inline(t):
    """行内处理：转义 + 行内代码 + 粗体 + 斜体"""
    t = _html.escape(t, quote=False)
    # 行内代码
    t = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", t)
    # 粗体
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    # 斜体（单星号，避免与粗体冲突）
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
    return t


def md_to_html(md):
    lines = md.split("\n")
    out = []
    i = 0
    in_code = False
    code_buf = []
    in_table = False
    table_rows = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>"); in_ul = False
        if in_ol:
            out.append("</ol>"); in_ol = False

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table:
            return
        # 过滤分隔行
        rows = [r for r in table_rows if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
        html = ["<table>"]
        for ri, r in enumerate(rows):
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            tag = "th" if ri == 0 else "td"
            html.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
        html.append("</table>")
        out.append("\n".join(html))
        in_table = False
        table_rows = []

    while i < len(lines):
        ln = lines[i]
        # 代码块
        if ln.strip().startswith("```"):
            if not in_code:
                in_code = True; code_buf = []
            else:
                in_code = False
                out.append("<pre><code>" + _html.escape("\n".join(code_buf)) + "</code></pre>")
            i += 1; continue
        if in_code:
            code_buf.append(ln); i += 1; continue
        # 表格
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            close_lists()
            in_table = True
            table_rows.append(ln)
            i += 1; continue
        else:
            flush_table()
        # 公式块 $$...$$
        if ln.strip().startswith("$$"):
            close_lists()
            buf = [ln.strip()]
            if ln.strip().endswith("$$") and len(ln.strip()) > 2:
                out.append(f"<div>{buf[0]}</div>")
            else:
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    buf.append(lines[i]); i += 1
                if i < len(lines):
                    buf.append(lines[i])
                out.append("<div>" + "\n".join(buf) + "</div>")
            i += 1; continue
        # 图片
        mimg = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln.strip())
        if mimg:
            close_lists()
            alt, src = mimg.group(1), mimg.group(2)
            b64 = fig_b64(os.path.basename(src))
            if b64:
                out.append(f'<img alt="{_html.escape(alt)}" src="{b64}">')
            else:
                out.append(f'<img alt="{_html.escape(alt)}" src="{src}">')
            i += 1; continue
        # 标题
        mh = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if mh:
            close_lists()
            lvl = len(mh.group(1))
            out.append(f"<h{lvl}>{inline(mh.group(2))}</h{lvl}>")
            i += 1; continue
        # 引用
        if ln.strip().startswith(">"):
            close_lists()
            content = ln.strip().lstrip(">").strip()
            buf = [content]; i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            out.append("<blockquote>" + " ".join(inline(b) for b in buf) + "</blockquote>")
            continue
        # 分割线
        if re.match(r"^\s*[-*_]{3,}\s*$", ln):
            close_lists()
            out.append("<hr>"); i += 1; continue
        # 无序列表
        mu = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if mu:
            if not in_ul:
                close_lists(); out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(mu.group(1))}</li>")
            i += 1; continue
        # 有序列表
        mo = re.match(r"^\s*\d+\.\s+(.*)$", ln)
        if mo:
            if not in_ol:
                close_lists(); out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(mo.group(1))}</li>")
            i += 1; continue
        # 空行
        if not ln.strip():
            close_lists()
            i += 1; continue
        # 普通段落
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
<title>问题二 完整解答报告 - F题</title>
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
{body}
</body>
</html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {OUT}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
