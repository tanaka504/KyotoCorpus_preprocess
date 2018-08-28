import os
import re
from pprint import pprint
import nltk
import json
import sys


# 正規表現のパターン
sid_pattern = re.compile(r"S-ID:(.*?)\s")
dst_pattern = re.compile(r"(.*?)[DPIA]")
coref_pattern = re.compile(r'=')
rel_pattern = re.compile(r'<rel (.*?)/>')
ne_pattern = re.compile(r'<ne (.*?)/>')
file_pattern = re.compile(r'^w201106-(.*?).KNP')
txt_file_pattern = re.compile(r'^(.*?).KNP$')

wdl_path = './kyoto_WDL_corpus/dat/rel/'
txt_path = './kyoto_txt_corpus/dat/rel'

# タグ単位のクラス
class Segment:
    def __init__(self):
        self.tag_id = -1
        self.tag_dst = -1
        self.label = ''
        self.wd_list = []
        self.corefs = []
        self.ne_word = []

    def __str__(self):
        return "{}\t{}".format(" ".join([str(wd[0]) for wd in self.wd_list]), ",".join(['|'.join(coref) for coref in self.corefs]))

	# extract coreference tag
    def coref_extract(self):
        labels = rel_pattern.findall(self.label)
        self.corefs = [(self.get_pattern('type', label),
						self.get_pattern('target', label),
						self.get_pattern('sid', label),
						self.get_pattern('tag', label))
					   for label in labels if coref_pattern.match(self.get_pattern('type', label)) if self.is_valid_tag(label)]
    
    def is_valid_tag(self, label):
        return False if self.get_pattern('type', label) == '' or self.get_pattern('target', label) == '' or self.get_pattern('sid', label) == '' or self.get_pattern('tag', label) == '' else True
        
    # extract ner tag
    def ne_extract(self):
        labels = ne_pattern.findall(self.label)
        self.ne_word = [(self.get_pattern('type', label),
						 self.get_pattern('target', label))
						for label in labels]
	# extract tag info with pattern matching
    def get_pattern(self, key, label):
        return re.search(r'{}=\"(.*?)\"'.format(key), label).group(1) if re.search(r'{}=\"(.*?)\"'.format(key), label) else ''

    # return start and end id
    def get_idx(self):
        return [self.wd_list[0][0], self.wd_list[-1][0]]

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

# タグ単位と文節単位で文情報を整理
def get_tags(data):
    wd_idx = 0
    segments = {}
    chunk = {}
    sentences = {}
    chunk_sent = []
    for idx, line in enumerate(data, 1):
        #print('processing line {} ...\n'.format(idx))
		# get sentence id
        if line[0] == "#":
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

		# get tag id dst
        elif line[0] == "+":
            tag_col = line.split(" ")
            tag_dst = int(dst_pattern.search(tag_col[2]).group(1))
            tag_list = " ".join(tag_col[3:])
            tag_id = tag_col[1]
            assert tag_id not in segments, "Invalid id in segments"
            if not tag_id in segments: segments[tag_id] = Segment()
            segments[tag_id].tag_id = tag_id
            segments[tag_id].tag_dst = tag_dst
            segments[tag_id].label = tag_list
            segments[tag_id].coref_extract()
            segments[tag_id].ne_extract()

		# save chunks to sentences
        elif line == "EOS":
			# assert len(segments) > 1, "Invalid process in segments"
            assert sid not in sentences.keys(), "Invalid value in sentences"
            sentences[sid] = {k:v for k,v in segments.items()}
            if len(chunk) > 0: chunk_sent.append(list(zip(*sorted(chunk.items(), key=lambda x:x[0])))[1])
            chunk.clear()
            segments.clear()
		
		# save word to segment
        else:
            col = line.split(" ")
			#assert len(sgmnt) < 1, "You must assign some idea about not belonging words"
            segments[tag_id].wd_list.append((wd_idx,col[0]))
            chunk[chunk_id].words.append({'wd_idx': wd_idx,
                'wd': col[0],
                'pos':col[3]})
            wd_idx += 1

    return sentences, chunk_sent

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
    

# 共参照タグの整理
def get_clusters(doc):
    cluster = {}

    for sentence in doc.values():
        for mention in sentence.values():
            for coref in mention.corefs:
                if '{}|{}'.format(coref[2], coref[3]) not in cluster:
                    cluster['{}|{}'.format(coref[2], coref[3])] = [doc[coref[2]][coref[3]].get_idx(), mention.get_idx()]
                else:
                    cluster['{}|{}'.format(coref[2], coref[3])].append(mention.get_idx())

    return cluster

# Output JSON file
# 京大コーパスで speaker はすべて著者となるので同一の speaker タグを付与
# doc_key は genre の特徴量に用いられる，京大コーパスの場合文書かテキストコーパスかの２種類のタグを付与
def finalize(doc, chunks, genre):
    doc_data = {}
    #sentences = [[word[1] for mention in sentence.values() for word in mention.wd_list] for sentence in doc.values()]
    sentences = [[word['wd'] for chunk in sentence for word in chunk.words ]for sentence in chunks]
    speakers = [['Speaker#1' for chunk in sentence for word in chunk.words] for sentence in chunks]
    cluster = get_clusters(doc)
    ner = [mention.get_idx() + [word[0]] for sentence in doc.values() for mention in sentence.values() for word in mention.ne_word if len(mention.ne_word) > 0]
    #parse = chunk_parser(chunks)
    parse = []

    doc_data['sentences'] = sentences
    doc_data['clusters'] = [v for v in cluster.values()]
    doc_data['ner'] = ner
    doc_data['genre'] = genre
    doc_data['constituents'] = parse
    doc_data['speakers'] = speakers


    #with open("./train_data/train.japanese.jsonlines", "w") as out_f:
    #    out_f.write(json.dumps(doc_data))
    #    out_f.write("\n")
    return doc_data

'''
---data construction---

doc = {sentence_id : {
					tag_id : Segment()
					...
					}
		...
		}
chunks = [
    (Chunk1, Chunk2, ...),
    ...
]
		
'''

def preprocessor(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read().split("\n")
        data.remove("")
        doc, chunks = get_tags(data)
        #[print(v) for sgmnt in doc.values() for v in sgmnt.values()]
        #[print(chunk) for chunk in chunks[0]]
    return doc, chunks

def main():
    with open('./train_data/all.japanese.jsonlines', 'w') as outfile:
        # 京大ウェブ文書リードコーパスの処理
        genre = 'wb/00'
        for i in range(25):
            dir_path = wdl_path + 'w201106-000{0:02d}'.format(i)
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and file_pattern.match(f)]
            for filename in files:
                print('{}/{}'.format(dir_path, filename))
                doc, chunks = preprocessor(os.path.join(dir_path, filename))
                doc_data = finalize(doc, chunks, genre)
                outfile.write(json.dumps(doc_data))
                outfile.write('\n')
        # 京大テキストコーパスの処理
        genre = 'tx/00'
        files = [f for f in os.listdir(txt_path) if os.path.isfile(os.path.join(txt_path, f)) and txt_file_pattern.match(f)]
        for filename in files:
            print('{}/{}'.format(txt_path, filename))
            doc, chunks = preprocessor(os.path.join(txt_path, filename))
            doc_data = finalize(doc, chunks, genre)
            outfile.write(json.dumps(doc_data))
            outfile.write('\n')

    print('Complete!')

if __name__ == "__main__":
    main()
