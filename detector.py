import os
import re
import json
from pprint import pprint

rel_pattern = re.compile(r'<rel (.*?)/>')
ne_pattern = re.compile(r'<ne ')
file_pattern = re.compile(r'^w201106-(.*?).KNP$')
coref_pattern = re.compile(r'=')
root_path = './kyoto_WDL_corpus/dat/rel/'
txt_path = './kyoto_txt_corpus/dat/rel/'

def execute(dir_path):
    files = os.listdir(dir_path)
    files = [f for f in files if os.path.isfile(os.path.join(dir_path,  f))]
    for filename in files:
        if not file_pattern.match(filename): continue
        #print('detecting {} ...\n'.format(os.path.join(dir_path, filename)))
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
            data = f.read().split('\n')
            data.remove('')
            #[print('{}line{}\n'.format(os.path.join(dir_path, filename), str(idx))) for idx, line in enumerate(data, 1) if ne_pattern.search(line)]
            #tag_list = {'{}|{}'.format(filename, idx):rel_pattern.findall(line) for idx, line in enumerate(data, 1) if rel_pattern.search(line)}
            #[print(fileline) for fileline, labels in tag_list.items() for label in labels if coref_pattern.match(find_pattern(label, 'type')) if find_pattern(label, 'target') == '' or find_pattern(label, 'sid') == '' or find_pattern(label, 'tag')]
            for idx, line in enumerate(data, 1):
                if not rel_pattern.search(line): continue
                labels = rel_pattern.findall(line)
                for label in labels:
                    if not coref_pattern.match(find_pattern(label, 'type')): continue
                    if find_pattern(label, 'target') == '' or find_pattern(label, 'sid') == '' or find_pattern(label, 'tag') == '':
                        print('{}/{} line {}'.format(dir_path, filename, idx))


def find_pattern(label, key):
    return re.search(r'{}=\"(.*?)\"'.format(key), label).group(1) if re.search(r'{}=\"(.*?)\"'.format(key), label) else ''
    

def detect():
	# detect WDL corpus
	for i in range(25):
		dir_path = root_path + 'w201106-000{0:02d}'.format(i)
		execute(dir_path)
	
	# detect txt corpus
	dir_path = txt_path
	execute(dir_path)

def data_viewer():
    with open('./train_data/train.japanese.jsonlines', 'r') as f:
        data = f.read().split('\n')
        json_data = json.loads(data[0])
        pprint(json_data)

if __name__ == "__main__":
	data_viewer()
