src_file=''
tgt_file=''

with open(src_file, 'r', encoding='utf8') as fd:
    src_data = fd.readlines()
with open(tgt_file, 'r', encoding='utf8') as fd:
    tgt_data = fd.readlines()

new_src_data = []
new_tgt_data = []

for i in range(len(src_data)):
    src_line = src_data[i].strip()
    tgt_line = tgt_data[i].strip()
    if src_line != '' and tgt_line != '':
        new_src_data.append(src_line)
        new_tgt_data.append(tgt_line)

src_wfile=''
tgt_wfile=''

with open(src_wfile, 'w', encoding='utf8') as fw:
    for line in new_src_data:
        fw.write(line+'\n')

with open(tgt_wfile, 'w', encoding='utf8') as fw:
    for line in new_tgt_data:
        fw.write(line+'\n')