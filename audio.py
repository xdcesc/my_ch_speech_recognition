import os
import pickle
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc



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
		fileObject = open(savepath, 'w')
		for ip in wav_files:
			fileObject.write(ip)
			fileObject.write('\n')
		fileObject.close()
		return wav_files


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


	def cmvn(self, wavlist='wav.scp', savepath='cmvn.dict', n_input=26):
		fileObject = open(wavlist, 'r')
		cmvn_dict = {}
		i=1
		for audio_filename in fileObject.readlines():
			audio_filename = audio_filename.strip('\n')
			fs, audio = wav.read(audio_filename)
			orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)
			train_inputs = orig_inputs[::2]
			train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
			cmvn_dict[audio_filename] = train_inputs
		featureObject = open(savepath, 'wb+')
		#featureObject.write(pickle.dump(mfcc_dict))
		pickle.dump(cmvn_dict,featureObject)
		featureObject.close()
		return cmvn_dict


if __name__ == '__main__':
	p = audier('E:\\Data\\primewords_md_2018_set1\\primewords_md_2018_set1\\audio_files\\0\\0a')
	p.getwavfile()
	p.mfcc()
	p.cmvn()