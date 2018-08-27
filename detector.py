import os
import re
from extractor import preprocessor

ne_pattern = re.compile(r'<ne ')
file_pattern = re.compile(r'^w201106-(.*?).KNP$')
root_path = './kyoto_WDL_corpus/dat/rel/'
txt_path = './kyoto_txt_corpus/dat/rel/'

def detect(dir_path):
	files = os.listdir(dir_path)
	files = [f for f in files if os.path.isfile(os.path.join(dir_path,  f))]
	for filename in files:
		if not file_pattern.match(filename): continue
		#print('detecting {} ...\n'.format(os.path.join(dir_path, filename)))
		with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
			data = f.read().split('\n')
			data.remove('')
			[print('{}line{}\n'.format(os.path.join(dir_path, filename), str(idx))) for idx, line in enumerate(data, 1) if ne_pattern.search(line)]
				
def detect():
	# detect WDL corpus
	for i in range(25):
		dir_path = root_path + 'w201106-000{0:02d}'.format(i)
		detect(dir_path)
	
	# detect txt corpus
	dir_path = txt_path
	detect(dir_path)

def main():
	i = 0
	dir_path = root_path + 'w201106-000{0:02d}'.format(i)
	files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and file_pattern.match(f)]
	preprocessor(os.path.join(dir_path, files[0]))

if __name__ == "__main__":
	main()
