import os, re, sys
import json
from pprint import pprint
import subprocess

# 正規表現のパターン
sid_pattern = re.compile(r"S-ID:(.*?)\s")
dst_pattern = re.compile(r"(.*?)[A-Z]")
label_pattern = re.compile(r'(.*?)=\"(.*?)\"')

josi_pattern = re.compile(r'[がはを]')

file_pattern = re.compile(r'^./train_data/raw_sentences/(.*?)/(.*?)/sentences_doc(.*?)$')

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

def get_chunks(data):
    wd_idx = 0
    sent_id = 0
    chunk = {}
    chunk_sent = []
    ga_dic = {}
    idx_dic = {}

    for idx, line in enumerate(data, 1):
		# get sentence id (BCCWJ コーパスでは意味を成さない)
        if line[0] == "#":
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
            labels = col[2].split(' ')
            for label in labels:
                if label_pattern.search(label):
                    if label_pattern.search(label).group(1) =='ga':
                        ga_dic[label_pattern.search(label).group(2)] = wd_idx
                    elif label_pattern.search(label).group(1) == 'id':
                        idx_dic[label_pattern.search(label).group(2)] = wd_idx

            # Chunk Process
            chunk[chunk_id].words.append({'wd_idx': wd_idx,
                'wd': col[0],
                'pos':col2[0]})
            wd_idx += 1
    
    zero_clusters = [(k, v) for k, v in ga_dic.items() if not k in idx_dic.keys()]
    if len(zero_clusters) > 1: print(zero_clusters)

    return chunk_sent

def get_ins_idx(raw_sentences, completed_sentences):
    align_idx = {}
    completed_clusters = []
    raw_idx = 0
    tmp_compl_id = []
    ins_idx = {}

    i = 0

    while i < len(completed_sentences):
        if raw_sentences[raw_idx] == completed_sentences[i]:
            if len(tmp_compl_id) > 1:
                completed_clusters.append([tmp_compl_id[0], tmp_compl_id[0]])
                tmp_compl_id = []
            align_idx[raw_idx] = i
            raw_idx += 1
            i += 1
        # raw word < completed word
        elif raw_sentences[raw_idx] in completed_sentences[i]:
            #input('{} {}'.format(raw_sentences[raw_idx],completed_sentences[i]))
            tmp_wd = raw_sentences[raw_idx]
            com_wd = completed_sentences[i]
            raw_loop = 1
            com_loop = 1
            while tmp_wd != com_wd:
                if com_wd in tmp_wd:
                    com_wd = com_wd + completed_sentences[i + com_loop]
                    com_loop += 1
                    continue
                tmp_wd = tmp_wd + raw_sentences[raw_idx + raw_loop]
                #input('{}|{}'.format(tmp_wd, com_wd))
                raw_loop += 1
            raw_idx += raw_loop
            i += com_loop
        # completed word < raw word
        elif completed_sentences[i] in raw_sentences[raw_idx]:
            #input('{} {}'.format(raw_sentences[raw_idx],completed_sentences[i]))
            tmp_wd = completed_sentences[i]
            raw_wd = raw_sentences[raw_idx]
            com_loop = 1
            raw_loop = 1
            while tmp_wd != raw_wd:
                if raw_wd in tmp_wd:
                    raw_wd = raw_wd + raw_sentences[raw_idx + raw_loop]
                    raw_loop += 1
                    continue
                tmp_wd = tmp_wd + completed_sentences[i + com_loop]
                #input('{}|{}'.format(tmp_wd, raw_wd))
                com_loop += 1
            i += com_loop
            raw_idx += raw_loop
        else:
            input('{} {}'.format(raw_sentences[raw_idx],completed_sentences[i]))
            if completed_sentences[i] == '人' and josi_pattern.match(completed_sentences[i+1]):
                assert not raw_idx in ins_idx, 'Unexpect idx in ins_idx'
                ins_idx[raw_idx] = ('人', completed_sentences[i+1])
            tmp_compl_id.append(i)
    #assert len(align_idx) == len(raw_sentences), 'Invalid align dict length'

    return ins_idx

def reindex(raw_sentences, completed_sentences):
    ins_idx = get_ins_idx(raw_sentences, completed_sentences)
    if len(ins_idx) > 1: print('completed sentence exist')


# adapt values to completed sentence
def finalize(json_data, compl_chunks, align_idx, zero_clusters):
    clusters = [[int(align_idx[word_idx]) for word_idx in cluster] for cluster in json_data['clusters']] + [zero_clusters]

    json_data['sentences'] = [[word['wd'] for chunk in sentence for word in chunk.words] for sentence in chunks]
    json_data['clusters'] = clusters
    
    return json_data

def main():
    with open('file_align') as align_f:
        align = align_f.read().split('\n')
        align.remove('')
        for num_iter, line in enumerate(align):
            line = line.split(' ')
            c = file_pattern.search(line[0]).group(1)
            p = file_pattern.search(line[0]).group(2)
            num = int(file_pattern.search(line[0]).group(3))
            
            with open('./train_data/no_zero/{}/{}.japanese.jsonlines'.format(c,p), 'r') as f, open('./train_data/sys_zero/{}/{}.japanese.jsonlines'.format(c, p), 'w') as out_f, open(line[1]) as compl_f:
                raw_data = f.read().split('\n')
                raw_jsondata = json.loads(raw_data[num])
                raw_sentences = [word for sentence in raw_jsondata['sentences'] for word in sentence]

                compl_data = compl_f.read().split('\n')
                compl_data.remove('')
                compl_chunks = get_chunks(compl_data)
                compl_sentences = [word['wd'] for sentence in compl_chunks for chunk in sentence for word in chunk.words]
                #compl_dir = './train_data/completed_sentences/{}/{}/'.format(c, p)
                #data = f.read().split('\n')
                
                #for idx, line in enumerate(data):
                #    json_data = json.loads(line)
                #    raw_sentences = [word for sentence in json_data['sentences'] for word in sentence]
                #    with open(os.path.join(compl_dir, 'completed_sentecne_doc{}'.format(idx)), 'r') as compl_f:
                #        compl_data = compl_f.read().split('\n')
                #        compl_data.remove('')
                #        compl_chunks = get_chunks(compl_data)
                #        compl_sentences = [word['wd'] for chunk in compl_chunks for word in chunk.words]
                #reindex(raw_sentences, compl_sentences)
                #doc_data = finalize(json_data, compl_chunks, align_idx, zero_clusters)
                
                #out_f.write(json.dumps(doc_data, ensure_ascii=False))
                #out_f.write('\n')
            if num_iter % 100 == 0:
                print('Complete {}/{} files'.format(num_iter, len(align)))

# jsonfiles -> flatten raw sentences per document
def json2sentences():
    for c in ['kyoto', 'ntc']:
        for p in ['train', 'dev', 'test']:
            with open('./train_data/no_zero/{}/{}.japanese.jsonlines'.format(c, p), 'r') as f:
                data = f.read().split('\n')
                data.remove('')
                for idx, line in enumerate(data):
                    jsondata = json.loads(line)
                    raw_sentence = '\n'.join([''.join([word for word in sentence])for sentence in jsondata['sentences']])
                    with open('./train_data/raw_sentences/{}/{}/sentences_doc{}'.format(c, p, idx), 'w') as out_f:
                        out_f.write(raw_sentence)

# test at 1 file
def test():
    with open('file_align') as fnames:
        fs = fnames.read().split('\n')
        ff = fs[56].split(' ')
        num = int(file_pattern.search(ff[0]).group(3))
        print(num)
    with open('./train_data/no_zero/ntc/train.japanese.jsonlines') as raw_f, open('./train_data/completed_sentences/ntc/train/completed_sentence_doc56') as compl_f:
        data = raw_f.read().split('\n')
        jsondata = json.loads(data[num])
        raw_sentences = [word for sentence in jsondata['sentences'] for word in sentence]
        print(jsondata['doc_key'])

        compl_data = compl_f.read().split('\n')
        compl_data.remove('')
        compl_chunks = get_chunks(compl_data)
        compl_sentences = [word['wd'] for sentence in compl_chunks for chunk in sentence for word in chunk.words]

        #_, _, ins_idx = reindex(raw_sentences, compl_sentences)
        #print(ins_idx)
        

if __name__=='__main__':
    main()
    #json2sentences()


