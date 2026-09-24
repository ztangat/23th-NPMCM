import zipfile, os, sys, io

base = r'C:\Users\dkyyt\Desktop\F题'
out = open(os.path.join(base, '_tools', 'probe_out.txt'), 'w', encoding='utf-8')
sys.stdout = out
sys.stderr = out

p = os.path.join(base, '算力约束下提升大语言模型能力的资源配置建模.docx')
z = zipfile.ZipFile(p)
print([n for n in z.namelist()])
print("=== dir listing ===")
for f in sorted(os.listdir(base)):
    fp = os.path.join(base, f)
    print(f, os.path.getsize(fp) if os.path.isfile(fp) else '<DIR>')
print("=== real_attachments ===")
ra = os.path.join(base, 'real_attachments')
for f in sorted(os.listdir(ra)):
    fp = os.path.join(ra, f)
    print(f, os.path.getsize(fp) if os.path.isfile(fp) else '<DIR>')
