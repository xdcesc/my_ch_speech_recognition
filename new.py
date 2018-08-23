import os
import numpy as np
import scipy.io.wavfile as wav
from collections import Counter
from python_speech_features import mfcc


def genwavlist(wavpath):
	wavfiles = {}
	fileids = []
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		print(dirpath)
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavfiles[fileid] = filepath
				fileids.append(fileid)
	return wavfiles,fileids

def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=13)
	return mfcc_feat

def gendict(textfile_path):
	dicts = []
	textfile = open(textfile_path,'r+')
	for content in textfile.readlines():
		content = content.strip('\n')
		content = content.split(' ',1)[1]
		content = content.split(' ')
		dicts += (word for word in content)
	counter = Counter(dicts)
	words = sorted(counter)
	wordsize = len(words)
	word2num = dict(zip(words, range(wordsize)))
	return word2num,len(word2num)

def text2num(textfile_path):
	lexcion,wordnum = gendict(textfile_path)
	word2num = lambda word:lexcion.get(word, 0)
	textfile = open(textfile_path, 'r+')
	content_dict = {}
	for content in textfile.readlines():
		content = content.strip('\n')
		cont_id = content.split(' ',1)[0]
		content = content.split(' ',1)[1]
		content = content.split(' ')
		content = list(map(word2num,content))
		add_num = list(np.zeros(50-len(content)))
		content = content + add_num
		content_dict[cont_id] = content
	return content_dict,lexcion

def get_batch(x, y, train=False, max_pred_len=4, input_length=8):
    X = np.expand_dims(x, axis=3)
    X = x # for model2
#     labels = np.ones((y.shape[0], max_pred_len)) *  -1 # 3 # , dtype=np.uint8
    labels = y
    input_length = np.ones([x.shape[0], 1]) * ( input_length - 2 )
#     label_length = np.ones([y.shape[0], 1])
    label_length = np.sum(labels > 0, axis=1)
    label_length = np.expand_dims(label_length,1)
    inputs = {'the_input': X,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              }
    outputs = {'ctc': np.zeros([x.shape[0]])}  # dummy data for dummy loss function
    return (inputs, outputs)


def data_generate(wavpath = 'E:\\Data\\data_thchs30\\dev', textfile = 'E:\\Data\\thchs30\\cv.syllable.txt', bath_size=32):
	wavdict,fileids = genwavlist(wavpath)
	content_dict,lexcion = text2num(textfile)
	genloop = len(fileids)//bath_size
	for i in range(genloop):
		print("the ",i,"'s loop")
		feats = []
		labels = []
		for x in range(bath_size):
			num = i * bath_size + x
			fileid = fileids[num]
			mfcc_feat = compute_mfcc(wavdict[fileid])
			feats.append(mfcc_feat)
			labels.append(content_dict[fileid])
		yield feats,labels






yields = data_generate()
for i,j in yields:
	print(np.shape(i))
	print(np.shape(j))
#text2num('E:\\Data\\thchs30\\cv.syllable.txt')