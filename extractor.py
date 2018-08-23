import re
from pprint import pprint

sid_pattern = re.compile(r"S-ID:(.*?)\sJUMAN")
dst_pattern = re.compile(r"(.*?)D")

class Segment:
	def __init__(self):
		self.tag_id = -1
		self.tag_dst = -1
		self.tag_list = ''
		self.wd_list = []
		self.chunk_info = None

	def __str__(self):
		return "{}\t{}".format(" ".join(wd_list), tag_list)
		
class Chunk:
	def __init__(self):
		self.chunk_id = -1
		self.chunk_dst = -1
		self.chunk_src = []
		self.segments = []
	
	def __str__(self):
		sentence = '|'.join([' '.join(sgmnt.wd_list) for sgmnt in self.segments])
		return '{}\tdst:{}\tsrcs:{}'.format(sentence, self.chunk_dst, self.chunk_src)

def get_docs(data):
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
			sgmnt = Segment()
			tag_col = line.split(" ")
			sgmnt.tag_dst = int(dst_pattern.search(tag_col[2]).group(1))
			sgmnt.tag_list = " ".join(tag_col[3:])
			tag_id = int(tag_col[1])
			sgmnt.tag_id = tag_id
			sgmnt.chunk_info = chunk
			print(tag_id, segments)
			assert tag_id in segments.keys(), "Invalid value in segments"
			segments[tag_id] = sgmnt

		# save chunks to sentences
		elif line == "EOS":
			assert len(chunks) > 1, "Invalid process in Chunks"
			assert sid in sentences.keys(), "Invalid value in sentences"
			sentences[sid] = segments
			segments.clear()
		
		# save word to segment
		else:
			col = line.split("\n")
			assert len(sgmnt) < 1, "You must assign some idea about not belonging words"
			sgmnt[tag_id].wd_list.append(col[0])
			
	return sentences

def main():
	with open("kyoto_WDL_corpus/dat/rel/w201106-00000/w201106-0000060050.KNP", "r") as f:
		data = f.read().split("\n")
		data.remove("")
		doc = get_docs(data)
		pprint(doc)	

if __name__ == "__main__":
	main()
