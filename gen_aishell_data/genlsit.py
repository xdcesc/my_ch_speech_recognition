import os

def gen_wavlist(wavpath,savefile):
	fileids = []
	fileObject = open(savefile, 'w+', encoding='UTF-8')
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				str1 =  ''
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				str1 = fileid + ' ' + filepath
				fileObject.write(str1 + '\n')
	fileObject.close()

gen_wavlist('data_aishell/wav/train','train.wav.lst')
