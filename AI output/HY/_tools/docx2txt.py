import zipfile, os, sys, re

base = r'C:\Users\dkyyt\Desktop\F题'
src = os.path.join(base, '算力约束下提升大语言模型能力的资源配置建模.docx')
dst = os.path.join(base, '_tools', 'problem_text.txt')
out = open(dst, 'w', encoding='utf-8')

z = zipfile.ZipFile(src)
xml = z.read('word/document.xml').decode('utf-8')

# split into paragraphs
paras = re.findall(r'<w:p\b[^>]*>(.*?)</w:p>', xml, re.S)
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def para_text(p):
    # runs
    texts = []
    for m in re.finditer(r'<w:t\b[^>]*>(.*?)</w:t>', p, re.S):
        t = m.group(1)
        t = t.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        texts.append(t)
    # drawings / images
    if re.search(r'<w:drawing\b', p):
        texts.append('[[图片]]')
    # math (omml) - try to get text
    for m in re.finditer(r'<m:t\b[^>]*>(.*?)</m:t>', p, re.S):
        texts.append('[M]' + m.group(1))
    return ''.join(texts)

for p in paras:
    t = para_text(p)
    t = t.strip()
    if t:
        out.write(t + '\n')

out.close()
print_lines = open(dst, encoding='utf-8').read()
z2 = open(os.path.join(base, '_tools', 'probe_out.txt'), 'w', encoding='utf-8')
z2.write('LINES=%d CHARS=%d\n' % (len(print_lines.splitlines()), len(print_lines)))
z2.close()
