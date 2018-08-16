import audio
import text
import numpy as np
#import acoustic_model


if __name__ == '__main__':
	#p = audio.audier('E:\\Data\\data_thchs30\\dev')
	#p.getwavfile()
	#cmvn_mat = p.cmvn()
	#cmvn_mat = np.array(cmvn_mat)
	#print(cmvn_mat)


	t = text.label('E:\\Data\\thchs30\\cv.syllable.txt')
	text_dict = t.gen_dict()
	label_mat = t.gen_num_label()
	print(label_mat.shape)

