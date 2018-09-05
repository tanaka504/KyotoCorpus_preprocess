import os
import re
from pprint import pprint
import nltk
import json
import sys
import codecs

from collections import Counter


# 正規表現のパターン
sid_pattern = re.compile(r"S-ID:(.*?)\s")
dst_pattern = re.compile(r"(.*?)[DPIA]")
coref_pattern = re.compile(r'=')
rel_pattern = re.compile(r'<rel (.*?)/>')
ne_pattern = re.compile(r'<ne (.*?)/>')

file_pattern = re.compile(r'^w201106-(.*?).KNP')
txt_file_pattern = re.compile(r'^(.*?).KNP$')


wdl_path = './data/KWDLC-1.0/dat/rel/'
txt_path = './data/KyotoCorpus4.0/dat/rel'

# タグ単位のクラス
class Segment:
    def __init__(self):
        self.sid = ''
        self.tag_id = -1
        self.tag_dst = -1
        self.label = ''
        self.wd_list = []
        self.corefs = []
        self.ne_word = []
        self.kakus = []

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

    def get_kaku(self):
        labels = rel_pattern.findall(self.label)
        self.kakus = [(self.get_pattern('type', label),self.get_pattern('sid', label), self.get_pattern('tag', label)) for label in labels if self.get_pattern('type', label) == 'ガ' or self.get_pattern('type', label) == 'ヲ']

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
    sent_idx = 0
    segments = {}
    chunk = {}
    sentences = {}
    chunk_sent = []
    for idx, line in enumerate(data, 1):
        #print('processing line {} ...\n'.format(idx))
		# get sentence id
        if line[0] == "#":
            sent_idx += 1
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
            segments[tag_id].sid = sent_idx
            segments[tag_id].tag_id = tag_id
            segments[tag_id].tag_dst = tag_dst
            segments[tag_id].label = tag_list
            segments[tag_id].coref_extract()
            segments[tag_id].ne_extract()
            segments[tag_id].get_kaku()

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

def get_srls(doc):
    srls = []

    for sentence in doc.values():
        for mention in sentence.values():
            for kaku in mention.kakus:
                if kaku[1] == '' or kaku[2] == '':
                    srls.append([mention.tag_id, -1, -1, kaku[0]])
                else:
                    sgmnt_idx = doc[kaku[1]][kaku[2]].get_idx()
                    srls.append([mention.tag_id, sgmnt_idx[0], sgmnt_idx[1], kaku[0]])

    return srls                


def distance_fleq(docs):
    cluster = {}
    result = {}

    for doc in docs:
        for sentence in doc.values():
            for mention in sentence.values():
                for coref in mention.corefs:
                    if (int(coref[2][-1]), int(coref[3])) not in cluster:
                        cluster[(int(coref[2][-1]), int(coref[3]))] = [int(mention.sid)]
                    else:
                        cluster[(int(coref[2][-1]), int(coref[3]))].append(int(mention.sid))
        c = Counter([k[0] - sid for k, v in cluster.items() for sid in v])
        for k, v in c.items():
            if k in result:
                result[k] += v
            else:
                result[k] = v
        
    
    [print('{}:{}'.format(k,v)) for (k, v) in sorted(result.items(), key= lambda x: -x[0])]


    

# Output JSON file
# speaker はすべて著者となるので同一の speaker タグを付与
# doc_key は genre の特徴量に用いられる，コーパスごとに(kw, kt, nt, bw)の4種類のタグを付与
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
    doc_data['doc_key'] = genre
    doc_data['constituents'] = parse
    doc_data['speakers'] = speakers


    #with open("./train_data/train.japanese.jsonlines", "w") as out_f:
    #    out_f.write(json.dumps(doc_data))
    #    out_f.write("\n")
    return doc_data


def preprocessor(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = f.read().split("\n")
        data.remove("")
        doc, chunks = get_tags(data)
        #[print(v) for sgmnt in doc.values() for v in sgmnt.values()]
        #[print(chunk) for chunk in chunks[0]]
    return doc, chunks

def preprocess_kyoto():
    with codecs.open('./train_data/all.kyoto_japanese.jsonlines', 'w', 'utf-8') as outfile:
        # 京大ウェブ文書リードコーパスの処理
        for i in range(25):
            dir_path = wdl_path + 'w201106-000{0:02d}'.format(i)
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and file_pattern.match(f)]
            for filename in files:
                genre = 'kw/{0:02d}/'.format(i) + filename
                try:
                    doc, chunks = preprocessor(os.path.join(dir_path, filename))
                    doc_data = finalize(doc, chunks, genre)
                except:
                    print('skip {}'.format(filename))
                    continue
                outfile.write(json.dumps(doc_data, ensure_ascii=False))
                outfile.write('\n')
    
        # 京大テキストコーパスの処理
    with codecs.open('./train_data/all.kyotxt_japanese.jsonlines','w', 'utf-8') as outfile:
        files = [f for f in os.listdir(txt_path) if os.path.isfile(os.path.join(txt_path, f)) and txt_file_pattern.match(f)]
        for filename in files:
            genre = 'kt/00/{}'.format(filename)
            try:
                doc, chunks = preprocessor(os.path.join(txt_path, filename))
                doc_data = finalize(doc, chunks, genre)
            except:
                print('skip {}'.format(filename))
                continue
            outfile.write(json.dumps(doc_data))
            outfile.write('\n')

    print('Complete kyoto preprocess!')

def test():
    sentence_len = 0
    sgmnt_len = 0
    coref_num = 0
    
    for i in range(25):
        dir_path = wdl_path + 'w201106-000{0:02d}'.format(i)
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and file_pattern.match(f)]
        for filename in files:
            genre = 'kw/{0:02d}/'.format(i) + filename
            try:
                doc, chunks = preprocessor(os.path.join(dir_path, filename))
                cluster = get_clusters(doc)
                sentence_len += len(doc)
                sgmnt_len += sum([len(sentence) for sentence in doc.values()])
                coref_num += sum([len(mentions) -1 for mentions in cluster.values()])
            except:
                print('skip {}'.format(filename))
                continue
    print(sentence_len, sgmnt_len, coref_num)
    sentence_len = 0
    sgmnt_len = 0
    coref_num = 0
    
    # 京大テキストコーパスの処理
    files = [f for f in os.listdir(txt_path) if os.path.isfile(os.path.join(txt_path, f)) and txt_file_pattern.match(f)]
    for filename in files:
        genre = 'kt/00/{}'.format(filename)
        try:
            doc, chunks = preprocessor(os.path.join(txt_path, filename))
            cluster = get_clusters(doc)
            sentence_len += len(doc)
            sgmnt_len += sum([len(sentence) for sentence in doc.values()])
            coref_num += sum([len(mentions) -1 for mentions in cluster.values()])
        except:
            print('skip {}'.format(filename))
            continue
    print(sentence_len, sgmnt_len, coref_num)
    
    print('Finish file reads')
    
    

def test_1file():
    dir_path = wdl_path + 'w201106-00000'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and file_pattern.match(f)]
    doc, chunks = preprocessor(os.path.join(dir_path, files[10]))
    #srls = get_srls(doc)
    c = get_clusters(doc)
    pprint(c)
    [print(len(mentions)-1) for mentions in c]

if __name__ == "__main__":
    #preprocess_kyoto()
    test()
