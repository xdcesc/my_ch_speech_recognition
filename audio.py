import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc


class audier(object):
	"""docstring for audier"""
	def __init__(self, wavepath):
		super(audier, self).__init__()
		self.wavepath = wavepath
	
	def getwavfile(self):
		'''获取音频文件列表'''
		wav_files = []
		for (dirpath, dirnames, filenames) in os.walk(self.wavepath):
			for filename in filenames:
				if filename.endswith('.wav') or filename.endswith('.WAV'):
					filename_path = os.sep.join([dirpath, filename])
					wav_files.append(filename_path)
		fileObject = open('wav.scp', 'w')
		for ip in wav_files:
			fileObject.write(ip)
			fileObject.write('\n')
		fileObject.close()


	def mfcc(self, wavlist='wav.scp', n_input=26):
		fileObject = open(wavlist, 'r')
		featureObject = open('wav.dict','w')
		for audio_filename in fileObject.readlines():
			audio_filename = audio_filename.strip('\n')
			fs, audio = wav.read(audio_filename)
			orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)

			print(fs)


if __name__ == '__main__':
	p = audier('E:\\Data\\data_thchs30\\data')
	p.getwavfile()
	p.mfcc()
