import zipfile, re, collections, sys

p = r"项目说明PPT\PPT过程\FSM_FSA_integrated_unified_principle_explanation.pptx"
z = zipfile.ZipFile(p)
names = z.namelist()
print("=== parts ===")
for n in sorted(names):
    if 'theme' in n or 'slideMaster' in n or 'slideLayout' in n or n.endswith('presentation.xml'):
        print(" ", n)

for tn in [n for n in names if re.match(r'ppt/theme/theme\d+\.xml$', n)]:
    xml = z.read(tn).decode('utf8', 'ignore')
    print("\n=== %s clrScheme ===" % tn)
    m = re.search(r'<a:clrScheme.*?</a:clrScheme>', xml, re.S)
    if m:
        for cm in re.finditer(r'<a:(\w+)>\s*<a:(?:srgbClr val="([0-9A-Fa-f]{6})"|sysClr[^/]*lastClr="([0-9A-Fa-f]{6})")', m.group(0)):
            print("  %-12s #%s" % (cm.group(1), cm.group(2) or cm.group(3)))
    fm = re.search(r'<a:fontScheme.*?</a:fontScheme>', xml, re.S)
    if fm:
        for fmm in re.finditer(r'<a:(majorFont|minorFont)>\s*<a:latin typeface="([^"]*)"', fm.group(0)):
            print("  %s latin: %s" % (fmm.group(1), fmm.group(2)))
        for fmm in re.finditer(r'<a:(majorFont|minorFont)>.*?<a:ea typeface="([^"]*)"', fm.group(0), re.S):
            print("  %s ea: %s" % (fmm.group(1), fmm.group(2)))

cnt = collections.Counter()
for n in names:
    if re.match(r'ppt/slides/slide\d+\.xml$', n):
        xml = z.read(n).decode('utf8', 'ignore')
        for c in re.findall(r'srgbClr val="([0-9A-Fa-f]{6})"', xml):
            cnt[c.upper()] += 1
print("\n=== top srgbClr across slides ===")
for c, k in cnt.most_common(30):
    print("  #%s  %d" % (c, k))

# font types used
fcnt = collections.Counter()
for n in names:
    if re.match(r'ppt/slides/slide\d+\.xml$', n):
        xml = z.read(n).decode('utf8', 'ignore')
        for c in re.findall(r'typeface="([^"]+)"', xml):
            fcnt[c] += 1
print("\n=== top typefaces across slides ===")
for c, k in fcnt.most_common(15):
    print("  %-28s %d" % (c, k))
