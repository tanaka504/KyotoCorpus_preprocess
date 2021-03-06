import os
import re
from pprint import pprint
import nltk
import json
import sys
import codecs
from collections import Counter
import utils

# 正規表現のパターン
sid_pattern = re.compile(r"S-ID:(.*?)\s")
dst_pattern = re.compile(r"(.*?)[DPIA]")
coref_pattern = re.compile(r'eq=\"(.*?)\"')
label_pattern = re.compile(r'(.*?)=\"(.*?)\"')

file_pattern = re.compile(r'^(.*?).ntc$')

ntc_path = './data/NTC_1.5/dat/ntc/knp-utf8/'

# 文節単位のクラス
class Chunk:
    def __init__(self):
        self.chunk_id = -1
        self.chunk_dst = -1
        self.chunk_src = []
        self.words = []

    def __str__(self):
        sentence = ' '.join([str(wd['wd']) for wd in self.words])
        return '{}\tdst:{}\tsrcs:{}'.format(sentence, self.chunk_dst, self.chunk_src)

    def get_idx(self):
        return [self.words[0]['wd_idx'], self.words[-1]['wd_idx']]

class NE:
    def __init__(self, ne):
        self.ne = ne
        self.wd = []

    def __str__(self):
        return '{}-{}|{}'.format(self.wd[0], self.wd[-1], self.ne)

    def get_idx(self):
        return [self.wd[0], self.wd[-1], self.ne]


# タグ単位と文節単位で文情報を整理
def get_tags(data):
    wd_idx = 0
    sent_id = 0
    ne = None
    corefs = {}
    distance = {}
    chunk = {}
    chunk_sent = []
    doc_srl = []
    nes = []
    zero_corefs = {}

    for idx, line in enumerate(data, 1):
        #print('processing line {} ...\n'.format(idx))
		# get sentence id
        if line[0] == "#":
            if sent_id > 0: doc_srl.append(get_srls(srls))
            srls = {}
            sent_id += 1
            sid = str(sid_pattern.search(line).group(1))
            
		
		# get chunk id dst
        elif line[0] == "*":
            chunk_col = line.split(" ")
            chunk_dst = int(dst_pattern.search(chunk_col[2]).group(1))
            chunk_id = int(chunk_col[1])
            if chunk_id not in chunk: chunk[chunk_id] = Chunk()
            chunk[chunk_id].chunk_dst = chunk_dst


            if not chunk_dst == -1:
                if chunk_dst not in chunk: chunk[chunk_dst] = Chunk()
                chunk[chunk_dst].chunk_src.append(chunk_id)

		# save chunks to sentences
        elif line == "EOS":
            #assert sid not in sentences.keys(), "Invalid value in sentences"
            if len(chunk) > 0: chunk_sent.append(list(zip(*sorted(chunk.items(), key=lambda x:x[0])))[1])
            chunk.clear()
		
		# save word to segment
        else:
            col = line.split(" ")

            # Named Entity Process
            #ne_tag = col[6].split('-')
            #if ne_tag[0] == 'B':
            #    ne = NE(ne_tag[1])
            #    ne.wd.append(wd_idx)
            #elif ne_tag[0] == 'I':
            #    assert isinstance(ne, NE), 'Unexpect NEtag (without Begin)'
            #    assert ne.ne == ne_tag[1], 'Unexpect NEtag (different Entity)'
            #    ne.wd.append(wd_idx)
            #    
            #else:
            #    if isinstance(ne, NE):
            #        nes.append(ne.get_idx())
            #        ne = None
            
            # Coreference Process
            if coref_pattern.search(col[7]):
                eq_id = coref_pattern.search(col[7]).group(1)
                if eq_id in corefs.keys(): corefs[eq_id].append(wd_idx)
                else: corefs[eq_id] = [wd_idx]

                if eq_id in distance: distance[eq_id].append(sent_id)
                else: distance[eq_id] = [sent_id]

            if not col[7] == '_':
                # ゼロ照応の場合，ゼロ代名詞を補完
                for tag in col[7].split("/"):
                    if not label_pattern.search(tag): continue
                    if label_pattern.search(tag).group(1) == 'ga' and re.match(r'exo', label_pattern.search(tag).group(2)):
                        if not 'exog' in corefs: corefs['exog'] = []
                        corefs['exog'].append(wd_idx)
                        chunk[chunk_id].words.append({'wd_idx': wd_idx,
                            'wd': '人',
                            'pos': '名詞',
                            })
                        chunk[chunk_id].words.append({'wd_idx': wd_idx + 1,
                            'wd': 'が',
                            'pos': '助詞',
                            })
                        wd_idx += 2

                # SRL Process
                assert not wd_idx in srls.keys(), 'Unexpect key in srls'
                srls[wd_idx] = {label_pattern.search(tag).group(1) : label_pattern.search(tag).group(2) for tag in col[7].split(" ")}
            
            # Chunk Process
            chunk[chunk_id].words.append({'wd_idx': wd_idx,
                'wd': col[0],
                'pos':col[3]})
            wd_idx += 1
    #c = Counter([sids[0] - sid for sids in distance.values() for sid in sids[1:]])
    #[print('{}:{}'.format(k,v)) for k, v in c.items()]
    clusters = [[[idx, idx] for idx in eq] for eq in corefs.values()]
    if 'exog' in corefs: zero_ant_clusters = [[idx, idx] for idx in corefs['exog']]
    else: zero_ant_clusters = []
    doc_srl.append(get_srls(srls))

    return chunk_sent, clusters, nes, doc_srl, zero_ant_clusters


def get_srls(srls):
    srl_list = []
    id2wd = {dst : wd_idx for wd_idx, labels in srls.items() for label, dst in labels.items() if label == 'id'}
    for wd_idx, labels in srls.items():
        for kaku, dst in labels.items():
            if kaku == 'ga':
                srl_list.append([wd_idx, wd_idx, wd_idx, u'V'])
                if re.match(r'exo', dst):
                    srl_list.append([wd_idx, 0, 0, u'GA'])
                elif dst in id2wd.keys():
                    srl_list.append([wd_idx, id2wd[dst], id2wd[dst], u'GA'])
                else:
                    pass
            elif kaku == 'o':
                srl_list.append([wd_idx, wd_idx, wd_idx, u'V'])
                if re.match(r'exo', dst):
                    srl_list.append([wd_idx, 0, 0, u'WO'])
                elif dst in id2wd.keys():
                    srl_list.append([wd_idx, id2wd[dst], id2wd[dst], u'WO'])
                else:
                    pass
            else:
                pass

    return srl_list


# Output JSON file
# 京大コーパスで speaker はすべて著者となるので同一の speaker タグを付与
# doc_key は genre の特徴量に用いられる，京大コーパスの場合文書かテキストコーパスかの２種類のタグを付与
def finalize(nes, clusters, chunks, genre, srls, zeros):
    doc_data = {}
    sentences = [[word['wd'] for chunk in sentence for word in chunk.words ]for sentence in chunks]
    speakers = [['Speaker#1' for chunk in sentence for word in chunk.words] for sentence in chunks]

    doc_data['sentences'] = sentences
    doc_data['clusters'] = clusters
    doc_data['ner'] = nes
    doc_data['doc_key'] = genre
    doc_data['constituents'] = []
    doc_data['speakers'] = speakers
    doc_data['srl'] = srls
    doc_data['zero_clusters'] = zeros


    #with open("./train_data/train.japanese.jsonlines", "w") as out_f:
    #    out_f.write(json.dumps(doc_data))
    #    out_f.write("\n")
    return doc_data

def preprocessor(filename, genre):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read().split("\n")
        data.remove("")
        chunks, clusters, ner, srls, zero_ants = get_tags(data)
        doc_data = utils.finalize(chunks, clusters, ner, srls, zero_ants, genre)
    return doc_data

def preprocess_ntc():
    with codecs.open('./train_data/all.ntc_japanese.jsonlines', 'w', 'utf-8') as outfile:
        files = [f for f in os.listdir(ntc_path) if os.path.isfile(os.path.join(ntc_path, f)) and file_pattern.match(f)]
        for filename in files:
            genre = 'nt/00/' + filename
            #print('{}/{}'.format(dir_path, filename))
            #print(genre, end='\n')
            try:
                doc_data = preprocessor(os.path.join(ntc_path, filename), genre)
            except:
                print('skip {}'.format(filename))
                continue
            outfile.write(json.dumps(doc_data, ensure_ascii=False))
            outfile.write('\n')

    print('Complete NTC preprocess!')

# test at 1 file
def test():
    files = [f for f in os.listdir(ntc_path) if os.path.isfile(os.path.join(ntc_path, f)) and file_pattern.match(f)]
    doc_data = preprocessor(os.path.join(ntc_path, '950112-0164-950112237.ntc'), '')
    #pprint(srls)
    print(files[10])
    pprint(doc_data['clusters'])
    print('--------------------------')
    pprint(doc_data['sentences'])

if __name__ == "__main__":
    test()
