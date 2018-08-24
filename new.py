import os
import random
import numpy as np
import scipy.io.wavfile as wav
from collections import Counter
from python_speech_features import mfcc
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape 
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta
from keras.layers.recurrent import GRU
from keras.preprocessing.sequence import pad_sequences

def genwavlist(wavpath):
	wavfiles = {}
	fileids = []
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavfiles[fileid] = filepath
				fileids.append(fileid)
	return wavfiles,fileids


def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
	mfcc_feat = mfcc_feat[::3]
	mfcc_feat = np.transpose(mfcc_feat)  
	mfcc_feat = pad_sequences(mfcc_feat, maxlen=500, dtype='float', padding='post', truncating='post').T
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


def get_batch(x, y, train=False, max_pred_len=50, input_length=500):
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


def data_generate(wavpath = 'E:\\Data\\data_thchs30\\train', textfile = 'E:\\Data\\thchs30\\train.syllable.txt', bath_size=4):
	wavdict,fileids = genwavlist(wavpath)
	#print(wavdict)
	content_dict,lexcion = text2num(textfile)
	genloop = len(fileids)//bath_size
	print("all loop :", genloop)
	while True:
		feats = []
		labels = []
		i = random.randint(0,genloop-1)
		for x in range(bath_size):
			num = i * bath_size + x
			fileid = fileids[num]
			print(wavdict[fileid])
			mfcc_feat = compute_mfcc(wavdict[fileid])
			feats.append(mfcc_feat)
			labels.append(content_dict[fileid])
		feats = np.array(feats)
		labels = np.array(labels)
		#print(np.shape(feats))
		#print(np.shape(labels))
		inputs, outputs = get_batch(feats, labels)
		print(np.shape(labels))
		yield inputs, outputs


def ctc_lambda(args):
	labels, y_pred, input_length, label_length = args
	y_pred = y_pred[:, :, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def creatModel():
	input_data = Input(name='the_input', shape=(500, 26))
	layer_h1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
	layer_h1 = Dropout(0.3)(layer_h1)
	layer_h2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h1)
	layer_h3_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', dropout=0.3)(layer_h2) # GRU
	layer_h3_2 = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.3)(layer_h2) # GRU
	layer_h3 = add([layer_h3_1, layer_h3_2])
	layer_h4 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h3)
	layer_h4 = Dropout(0.3)(layer_h4)
	layer_h5 = Dense(1200, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h4)
	output = Activation('softmax', name='Activation0')(layer_h5)
	model_data = Model(inputs=input_data, outputs=output)
	#ctc
	labels = Input(name='the_labels', shape=[50], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, output, input_length, label_length])
	model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
	model.summary()
	ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
	#model=multi_gpu_model(model,gpus=2)
	model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=ada_d)
	#test_func = K.function([input_data], [output])
	print("model compiled successful!")
	return model, model_data


model, model_data = creatModel()
yielddatas = data_generate()
model.fit_generator(yielddatas,9600)
model.save_weights('model.mdl')
model_data.save_weights('model_data.mdl')
#text2num('E:\\Data\\thchs30\\cv.syllable.txt')