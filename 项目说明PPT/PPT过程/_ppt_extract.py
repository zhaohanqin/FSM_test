import zipfile, re, sys, os
from xml.etree import ElementTree as ET

NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
}

path = sys.argv[1]
out = sys.argv[2]

z = zipfile.ZipFile(path)
names = [n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)]
names.sort(key=lambda n: int(re.search(r'slide(\d+)\.xml', n).group(1)))

parts = []
for n in names:
    idx = int(re.search(r'slide(\d+)\.xml', n).group(1))
    root = ET.fromstring(z.read(n))
    lines = []
    # iterate shapes in document order, extract text frames
    for sp in root.iter():
        tag = sp.tag.split('}')[-1]
        if tag in ('txBody',):
            paras = []
            for p in sp.findall('a:p', NS):
                texts = [t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t')]
                line = ''.join(texts).strip()
                if line:
                    paras.append(line)
            if paras:
                lines.append('\n'.join(paras))
    # notes
    notes = None
    notesname = 'ppt/notesSlides/notesSlide%d.xml' % idx
    if notesname in z.namelist():
        nroot = ET.fromstring(z.read(notesname))
        nl = []
        for tb in nroot.iter():
            if tb.tag.split('}')[-1] == 'txBody':
                for p in tb.findall('a:p', NS):
                    texts = [t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t')]
                    line = ''.join(texts).strip()
                    if line:
                        nl.append(line)
        if nl:
            notes = '\n'.join(nl)
    # rels: count images
    relname = 'ppt/slides/_rels/slide%d.xml.rels' % idx
    imgs = []
    if relname in z.namelist():
        rroot = ET.fromstring(z.read(relname))
        for rel in rroot:
            tgt = rel.get('Target', '')
            if 'media' in tgt:
                imgs.append(tgt.split('/')[-1])
    parts.append('===== SLIDE %d =====\n[TEXT]\n%s\n[IMAGES] %s\n[NOTES] %s\n' % (
        idx, '\n---\n'.join(lines) or '(no text)', ', '.join(imgs) or '(none)', notes or '(none)'))

with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(parts))
print('slides:', len(names), 'chars:', sum(len(p) for p in parts))
