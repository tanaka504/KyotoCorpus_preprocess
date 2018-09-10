import os
import re
from pprint import pprint
import nltk
import json
import sys
import codecs
from ntc_preprocess import get_srls

# 正規表現のパターン
sid_pattern = re.compile(r"S-ID:(.*?)\s")
dst_pattern = re.compile(r"(.*?)[A-Z]")
coref_pattern = re.compile(r'eq=\"(.*?)\"')
label_pattern = re.compile(r'(.*?)=\"(.*?)\"')

file_pattern = re.compile(r'^(.*?).cabocha$')

bccwj_path = './data/BCCWJ/ntc/'

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
    nes = []
    chunk = {}
    chunk_sent = []
    doc_srl = []

    for idx, line in enumerate(data, 1):
        #print('processing line {} ...\n'.format(idx))
		# get sentence id (BCCWJ コーパスでは意味を成さない)
        if line[0] == "#":
            if sent_id > 0: doc_srl.append(get_srls(srls))
            srls = {}
            sent_id += 1
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
            col = line.split("\t")
            col2 = col[1].split(',')
            # Named Entity Process (BCCWJ コーパスには固有表現タグがない)
            #ne_tag = col[6].split('-')
            #if ne_tag[0] == 'B':
            #    ne = NE(ne_tag[1])
            #    ne.wd.append(wd_idx)
            #elif ne_tag[0] == 'I':
            #    assert isinstance(ne, NE), 'Unexpect NEtag (without Begin)'
            #    assert ne.ne == ne_tag[1], 'Unexpect NEtag (different Entity)'
            #    ne.wd.append(wd_idx)
                
            #else:
            #    if isinstance(ne, NE):
            #        nes.append(ne.get_idx())
            #        ne = None
            
            # Coreference Process
            if coref_pattern.search(col[2]):
                eq_id = coref_pattern.search(col[2]).group(1)
                if eq_id in corefs.keys(): corefs[eq_id].append(wd_idx)
                else: corefs[eq_id] = [wd_idx]

            if not col[2] == '_':
                assert not wd_idx in srls.keys(), 'Unexpect key in srls'
                srls[wd_idx] = {label_pattern.search(tag).group(1) : label_pattern.search(tag).group(2) for tag in col[2].split(" ")}

            # Chunk Process
            chunk[chunk_id].words.append({'wd_idx': wd_idx,
                'wd': col[0],
                'pos':col2[0]})
            wd_idx += 1
    
    clusters = [[[idx, idx] for idx in eq] for eq in corefs.values()]
    doc_srl.append(get_srls(srls))

    return nes, clusters, chunk_sent, doc_srl


# 係り元の根の文節番号を再帰的に導出
def get_root(sentence, idx, src):
    if len(src) < 1: return idx

    else:
        return min([get_root(sentence, s, sentence[s].chunk_src) for s in src])

def chunk_parser(sentences):
    # JUMANの品詞体系より
    pos_dict = {
            '形容詞': 'ADJ',
            '連体詞': 'ADN',
            '副詞': 'ADV',
            '判定詞': 'JUDG',
            '助動詞': 'AUXV',
            '接続詞': 'CONJ',
            '指示詞': 'DEMO',
            '感動詞': 'INTJ',
            '名詞': 'NOUN',
            '動詞': 'VERB',
            '助詞': 'PSTP',
            '接頭辞': 'PREF',
            '接尾辞': 'SUFF',
            '特殊': 'SPE',
            '未定義語': 'UNK',
            }
    # 1行で書いてて読みづらいですが，文の最後の文節から順番に係り元の左端の文節番号を取得し，
    # 係っている文節の始めと最後の単語のインデックス及び，右端の文節の始めの単語のPOSを取得
    roots = [[sentence[get_root(sentence, i-1, sentence[i-1].chunk_src)].get_idx()[0],sentence[i-1].get_idx()[1], pos_dict[sentence[i-1].words[0]['pos']]] for sentence in sentences for i in range(len(sentence), 0 ,-1)]

    return roots
    

# Output JSON file
# 京大コーパスで speaker はすべて著者となるので同一の speaker タグを付与
# doc_key は genre の特徴量に用いられる，京大コーパスの場合文書かテキストコーパスかの２種類のタグを付与
def finalize(nes, clusters, chunks, genre, srls):
    doc_data = {}
    #sentences = [[word[1] for mention in sentence.values() for word in mention.wd_list] for sentence in doc.values()]
    sentences = [[word['wd'] for chunk in sentence for word in chunk.words ]for sentence in chunks]
    speakers = [['Speaker#1' for chunk in sentence for word in chunk.words] for sentence in chunks]
    #parse = chunk_parser(chunks)
    parse = []

    doc_data['sentences'] = sentences
    doc_data['clusters'] = clusters
    doc_data['ner'] = nes
    doc_data['doc_key'] = genre
    doc_data['constituents'] = parse
    doc_data['speakers'] = speakers
    doc_data['srl'] = srls


    #with open("./train_data/train.japanese.jsonlines", "w") as out_f:
    #    out_f.write(json.dumps(doc_data))
    #    out_f.write("\n")
    return doc_data

def preprocessor(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read().split("\n")
        data.remove("")
        nes, clusters, chunks, srls = get_tags(data)
        #[print(v) for sgmnt in doc.values() for v in sgmnt.values()]
        #[print(chunk) for chunk in chunks[0]]
    return nes, clusters, chunks, srls

def preprocess_bccwj():
    with codecs.open('./train_data/all.bccwj_japanese.jsonlines', 'w', 'utf-8') as outfile:
        files = [f for f in os.listdir(bccwj_path) if os.path.isfile(os.path.join(bccwj_path, f)) and file_pattern.match(f)]
        for filename in files:
            genre = 'bw/00/' + filename
            #print('{}/{}'.format(dir_path, filename))
            #print(genre, end='\n')
            try:
                nes, clusters, chunks, srls = preprocessor(os.path.join(bccwj_path, filename))
                doc_data = finalize(nes, clusters, chunks, genre, srls)
            except:
                print('skip {}'.format(filename))
                continue
            if len(['' for sentence in doc_data['sentences'] for word in sentence]) > 4000: continue
            outfile.write(json.dumps(doc_data, ensure_ascii=False))
            outfile.write('\n')

    print('Complete BCCWJ preprocess!')

# test at 1 file
def test():
    #files = [f for f in os.listdir(bccwj_path) if os.path.isfile(os.path.join(bccwj_path, f)) and file_pattern.match(f)]
    nes, clusters, chunks, srls = preprocessor(os.path.join(bccwj_path, '00012_A_PN4g_00001.cabocha'))
    doc_data = finalize(nes, clusters, chunks, '', srls)
    #pprint(nes)
    pprint(srls)
    pprint(clusters)
    print(len(chunks))


if __name__ == "__main__":
    #preprocess_bccwj()
    test()
