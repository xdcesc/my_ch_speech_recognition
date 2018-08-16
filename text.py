import numpy as np
import pickle
from collections import Counter


class label(object):
	"""docstring for label"""
	def __init__(self, textpath):
		super(label, self).__init__()
		self.textpath = textpath


	def dict_test(self):
		all_word = []
		label = []
		# python字典生成方法，没有语音，只生成汉字对应了序号
		label = ['hello','hello','hello','hi','hi','hi','you', 'you']
		all_word += [word for word in label]
		counter = Counter(all_word)
		words = sorted(counter)
		words_size = len(words)
		word_num_map = dict(zip(words, range(words_size)))
		to_num = lambda word: word_num_map.get(word, 4)
		label_vector = list(map(to_num,['hello','you','hi','you']))
		print(label_vector)
		print('\nall words i use :\n',all_word)
		print('\nafter counter:\n',counter)
		print('\nafter sort:\n',words)
		print('\nthe words size: ',words_size)
		print('\nthe words num map:\n',word_num_map)


	def gen_dict(self, savepath='text2num.dict'):
		textlabels = []
		textobj = open(self.textpath, 'r+')
		for textlabel in textobj.readlines():
			textlabel = textlabel.strip('\n')
			textlabel_id = textlabel.split(' ',1)[0]
			textlabel_text = textlabel.split(' ',1)[1]
			textlabel_text = textlabel_text.split(' ')
			textlabels += (word for word in textlabel_text)
		counter = Counter(textlabels)
		words = sorted(counter)
		words_size = len(words)
		word2num = dict(zip(words, range(words_size)))
		dictobj = open(savepath, 'wb+')
		#featureObject.write(pickle.dump(mfcc_dict))
		pickle.dump(word2num,dictobj)
		dictobj.close()
		to_num = lambda word: word2num.get(word, 4)
		print(len(word2num))
		return word2num


	def gen_num_label(self, savepath='num_label.dict'):
		word2num = self.gen_dict()
		to_num = lambda word: word2num.get(word, 4)
		textobj = open(self.textpath, 'r+')
		
		labeldict = {}
		for textlabel in textobj.readlines():
			textlabel = textlabel.strip('\n')
			textlabel_id = textlabel.split(' ',1)[0]
			textlabel_text = textlabel.split(' ',1)[1]
			textlabel_text = textlabel_text.split(' ')
			label_vector = list(map(to_num,textlabel_text))
			labeldict[textlabel_id] = label_vector
		labelobj = open(savepath, 'wb+')
		pickle.dump(labeldict,labelobj)
		labelobj.close()
		print(labeldict)
		return labeldict


if __name__ == '__main__':
	t = label('E:\\Data\\thchs30\\train.syllable.txt')
	t.gen_dict()
	t.gen_num_label()