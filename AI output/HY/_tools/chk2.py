import sys, importlib
out = open(r'C:\Users\dkyyt\Desktop\F题\_tools\chk2_out.txt', 'w', encoding='utf-8')
out.write(sys.executable + '\n')
out.write(sys.version + '\n')
for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'matplotlib', 'pyarrow', 'lzma', 'json']:
    try:
        mod = importlib.import_module(m)
        out.write('%s = %s\n' % (m, getattr(mod, '__version__', 'ok')))
    except Exception as e:
        out.write('%s MISSING (%s)\n' % (m, e))
out.close()
