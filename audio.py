import os
import pickle
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences


class audier(object):
	"""docstring for audier"""
	def __init__(self, wavepath):
		super(audier, self).__init__()
		self.wavepath = wavepath


	def getwavfile(self, savepath='wav.scp'):
		'''获取音频文件列表'''
		wav_files = []
		for (dirpath, dirnames, filenames) in os.walk(self.wavepath):
			for filename in filenames:
				if filename.endswith('.wav') or filename.endswith('.WAV'):
					filename_path = os.sep.join([dirpath, filename])
					wav_files.append(filename_path)
		wav_files.sort()
		fileObject = open(savepath, 'w')
		for ip in wav_files:
			fileObject.write(ip)
			fileObject.write('\n')
		fileObject.close()
		return wav_files

	# 生成mfcc特征文件，可以通过文件路径名使用特征
	def mfcc(self, wavlist='wav.scp', savepath='mfcc.dict', n_input=26):
		fileObject = open(wavlist, 'r')
		mfcc_dict = {}
		i=1
		for audio_filename in fileObject.readlines():
			audio_filename = audio_filename.strip('\n')
			fs, audio = wav.read(audio_filename)
			orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)
			train_inputs = orig_inputs[::2]

			mfcc_dict[audio_filename] = train_inputs
		featureObject = open(savepath, 'wb+')
		#featureObject.write(pickle.dump(mfcc_dict))
		pickle.dump(mfcc_dict,featureObject)
		featureObject.close()
		return mfcc_dict

	#生成cmvn,可以通过文件名使用特征
	def cmvn(self, wavlist='wav.scp', savepath='cmvn.dict', n_input=26):
		fileObject = open(wavlist, 'r')
		cmvn_dict = {}
		cmvn_mat = []
		for audio_filename in fileObject.readlines():
			audio_filename = audio_filename.strip('\n')
			fs, audio = wav.read(audio_filename)
			orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)
			train_inputs = orig_inputs[::3]
			train_inputs = (train_inputs - np.mean(train_inputs,0)) / np.std(train_inputs,0)
			train_inputs = np.transpose(train_inputs)  
			train_inputs = pad_sequences(train_inputs, maxlen=500, dtype='float', padding='post', truncating='post').T
			#cmvn_dict[audio_filename] = train_inputs
			cmvn_mat.append(train_inputs)
		#featureObject = open(savepath, 'wb+')
		#featureObject.write(pickle.dump(mfcc_dict))
		#pickle.dump(cmvn_dict,featureObject)
		#featureObject.close()
		return cmvn_mat


if __name__ == '__main__':
	p = audier('E:\\Data\\data_thchs30\\dev')
	p.getwavfile()
	cmvn_mat = p.cmvn()
	cmvn_mat = np.array(cmvn_mat)
	print(cmvn_mat.shape)