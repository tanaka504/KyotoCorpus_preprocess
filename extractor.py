import re
from pprint import pprint

sid_pattern = re.compile(r"S-ID:(.*?)\sJUMAN")
dst_pattern = re.compile(r"(.*?)D")
coref_pattern = re.compile(r'=')
rel_pattern = re.compile(r'<rel (.*?)/>')
ne_pattern = re.compile(r'<ne (.*?)/>')

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
					   for label in labels if coref_pattern.match(self.get_pattern('type', label))]

	def ne_extract(self):
		labels = ne_pattern.findall(self.label)
		self.ne_word = [(self.get_pattern('type', label),
						 self.get_pattern('target', label))
						for label in labels]
	# extract tag info with pattern matching
	def get_pattern(self, key, label):
		return re.search(r'{}=\"(.*?)\"'.format(key), label).group(1)

	def get_idx(self):
		return [self.wd_list[0][0], self.wd_list[-1][0]]

class Chunk:
	def __init__(self):
		self.chunk_id = -1
		self.chunk_dst = -1
		self.chunk_src = []

	def __str__(self):
		sentence = '|'.join([' '.join(sgmnt.wd_list) for sgmnt in self.segments])
		return '{}\tdst:{}\tsrcs:{}'.format(sentence, self.chunk_dst, self.chunk_src)

def get_tags(data):
	wd_idx = 0
	segments = {}
	sentences = {}
	for line in data:
		# get sentence id
		if line[0] == "#":
			sid = str(sid_pattern.search(line).group(1))
		
		# get chunk id dst
		elif line[0] == "*":
			chunk = Chunk()
			chunk_col = line.split(" ")
			chunk.chunk_dst = int(dst_pattern.search(chunk_col[2]).group(1))
			chunk.chunk_id = int(chunk_col[1])

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
			segments.clear()
		
		# save word to segment
		else:
			col = line.split(" ")
			#assert len(sgmnt) < 1, "You must assign some idea about not belonging words"
			segments[tag_id].wd_list.append((wd_idx,col[0]))
			wd_idx += 1

	return sentences

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
def finalize(doc):
	doc_data = {}
	sentences = [[word[1] for mention in sentence.values() for word in mention.wd_list] for sentence in doc.values()]
	cluster = get_clusters(doc)
	ner = [mention.get_idx() + [word[0]] for sentence in doc.values() for mention in sentence.values() for word in mention.ne_word if len(mention.ne_word) > 0]

	doc_data['sentences'] = sentences
	doc_data['clusters'] = [v for v in cluster.values()]
	doc_data['ner'] = ner


'''
---data construction---

doc = {sentence_id : {
					tag_id : Segment()
					...
					}
		...
		}
		
'''

def main():
	with open("kyoto_WDL_corpus/kyoto_test", "r", encoding='utf-8') as f:
		data = f.read().split("\n")
		data.remove("")
		doc = get_tags(data)
		[print(v) for sgmnt in doc.values() for v in sgmnt.values()]
		finalize(doc)


if __name__ == "__main__":
	main()