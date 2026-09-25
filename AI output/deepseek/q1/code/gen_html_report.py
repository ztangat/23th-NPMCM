# -*- coding: utf-8 -*-
"""
gen_html_report.py —— 将问题一 markdown 报告转为可离线浏览的 HTML（内嵌图片 base64）

读取：reports/问题一_完整解答报告.md
产出：reports/问题一_完整解答报告.html
依赖：markdown（若无则降级为极简转换）
"""
import os, sys, re, base64, html as _html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MD = os.path.join(REPORTS, "问题一_完整解答报告.md")
OUT = os.path.join(REPORTS, "问题一_完整解答报告.html")


def fig_b64(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


CSS = """
:root{--fg:#1a1a1a;--bg:#ffffff;--muted:#5b6472;--accent:#2563eb;--border:#e3e8ef;--code:#f6f8fa;}
*{box-sizing:border-box;}
body{font-family:"Segoe UI","Microsoft YaHei",-apple-system,sans-serif;line-height:1.75;
color:var(--fg);background:var(--bg);max-width:980px;margin:0 auto;padding:40px 28px 80px;}
h1{font-size:1.9em;border-bottom:3px solid var(--accent);padding-bottom:12px;margin-top:8px;}
h2{font-size:1.45em;margin-top:2.2em;border-left:5px solid var(--accent);padding-left:12px;}
h3{font-size:1.18em;margin-top:1.7em;color:#1f2937;}
h4{font-size:1.05em;color:#374151;}
table{border-collapse:collapse;width:100%;margin:1.1em 0;font-size:0.92em;}
th,td{border:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top;}
th{background:#f0f4fa;font-weight:600;}
tr:nth-child(even) td{background:#fafbfd;}
code{background:var(--code);padding:2px 6px;border-radius:4px;font-family:Consolas,Monaco,monospace;
font-size:0.9em;color:#b91c1c;}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;border:1px solid var(--border);}
blockquote{border-left:4px solid #94a3b8;margin:1em 0;padding:6px 16px;color:var(--muted);background:#f8fafc;}
hr{border:none;border-top:1px solid var(--border);margin:2.4em 0;}
img{max-width:100%;border:1px solid var(--border);border-radius:8px;margin:12px 0;
box-shadow:0 1px 6px rgba(0,0,0,.06);}
strong{color:#0f172a;}
a{color:var(--accent);}
.fig-tag{background:#eef4ff;border:1px solid #c7d8ff;border-radius:8px;padding:14px 16px;margin:14px 0;}
.fig-tag img{margin:6px 0 2px;}
.toc{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:14px 22px;margin:20px 0;}
.toc a{text-decoration:none;}
"""

# 简单 markdown → html（针对本报告结构：标题/表格/列表/引用/代码/公式原样保留）
def md_to_html(md):
    lines = md.split("\n")
    out = []
    in_table = False; in_code = False; in_ul = False; in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul: out.append("</ul>"); in_ul = False
        if in_ol: out.append("</ol>"); in_ol = False

    for ln in lines:
        # 代码块
        if ln.strip().startswith("```"):
            close_lists()
            if in_table: out.append("</table>"); in_table = False
            if not in_code:
                out.append("<pre><code>"); in_code = True
            else:
                out.append("</code></pre>"); in_code = False
            continue
        if in_code:
            out.append(_html.escape(ln)); continue

        s = ln.strip()
        # 表格
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # 分隔行
            if not in_table:
                close_lists()
                out.append("<table>"); in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            close_lists()
            lvl = len(m.group(1)); out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); continue
        # 分隔线
        if re.match(r"^-{3,}$", s):
            close_lists(); out.append("<hr>"); continue
        # 引用
        if s.startswith(">"):
            close_lists(); out.append(f"<blockquote>{inline(s[1:].strip())}</blockquote>"); continue
        # 列表
        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            if not in_ol:
                if in_ul: out.append("</ul>"); in_ul = False
                out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(m.group(2))}</li>"); continue
        if re.match(r"^[-*]\s+", s):
            if not in_ul:
                if in_ol: out.append("</ol>"); in_ol = False
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(re.sub(r'^[-*]\s+','',s))}</li>"); continue
        close_lists()
        if s == "":
            out.append("")
        else:
            out.append(f"<p>{inline(s)}</p>")
    if in_table: out.append("</table>")
    close_lists()
    return "\n".join(out)


def inline(t):
    t = _html.escape(t, quote=False)
    # 行内代码
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    # 粗体
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    # 图片标记 [[FIG:name]]
    t = re.sub(r"\[\[FIG:([^\]]+)\]\]", lambda m: fig_tag(m.group(1)), t)
    return t


def fig_tag(name):
    b = fig_b64(name)
    if b is None:
        return f'<em>[缺图: {name}]</em>'
    return f'<div class="fig-tag"><img src="{b}" alt="{name}"><div style="font-size:.9em;color:#5b6472">{name}</div></div>'


def main():
    md = open(MD, encoding="utf-8").read()
    body = md_to_html(md)
    # 图表专区：内嵌全部 8 张图，插入到"附录"标题之前
    figs = sorted(os.listdir(FIG))
    gallery = ['<h2 id="figs">图表专区（问题一全部产出图表）</h2>']
    for f in figs:
        gallery.append(fig_tag(f))
    gallery_html = "\n".join(gallery)
    if "<h2>附录" in body:
        body = body.replace("<h2>附录", gallery_html + "\n<hr>\n<h2>附录", 1)
    else:
        body = body + "\n<hr>\n" + gallery_html
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题一完整解答报告 · F题</title>
<style>{CSS}</style></head>
<body>
{body}
</body></html>"""
    open(OUT, "w", encoding="utf-8").write(html_doc)
    print(f"[OK] {OUT}  ({len(html_doc)//1024} KB)")


if __name__ == "__main__":
    main()
