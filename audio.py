import os
import pickle
import scipy.io.wavfile as wav
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


	def mfcc(self, wavlist='wav.scp', savepath='mfcc.dict', n_input=26):
		fileObject = open(wavlist, 'r')
		mfcc_dict = {}
		i=1
		for audio_filename in fileObject.readlines():
			audio_filename = audio_filename.strip('\n')
			fs, audio = wav.read(audio_filename)
			orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)
			orig_inputs = orig_inputs[::2]
			mfcc_dict[audio_filename] = orig_inputs
			if i%50==0:
				print('this is file:', i)
			i = i + 1
		featureObject = open(savepath, 'w')
		#featureObject.write(pickle.dump(mfcc_dict))
		pickle.dump(mfcc_dict,featureObject)
		featureObject.close()


if __name__ == '__main__':
	p = audier('E:\\Data\\data_thchs30\\data')
	p.getwavfile()
	p.mfcc()