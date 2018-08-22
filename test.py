import audio
import text
import acoustic_model
import numpy as np
import os
#import acoustic_model


if __name__ == '__main__':
	#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #　选择使用的GPU

	p = audio.audier('E:\\Data\\data_thchs30\\dev')
	p.getwavfile()
	cmvn_mat = p.cmvn()
	cmvn_mat = np.array(cmvn_mat)
	print(cmvn_mat.shape)

	t = text.label('E:\\Data\\thchs30\\cv.syllable.txt')
	text_dict = t.gen_dict()
	label_mat = t.gen_num_label()
	print(label_mat.shape)

	am = acoustic_model.speech_rnn(cmvn_mat, label_mat)
	print('success import amodel')
	am.LoadModel()
	basepre = am.TestModel()
	for i in range(1,100):
		print(basepre[0][i])