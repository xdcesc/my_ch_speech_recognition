import json
import numpy as np
import re
from pypinyin import pinyin, lazy_pinyin, Style





def trans_aishell_to_pinyin(word_path, pinyin_path):
	# 需要转换为拼音的中文汉字路径
	textobj = open(word_path, 'r+', encoding='UTF-8')
	# 转化为拼音后的保存txt路径
	savefile = open(pinyin_path, 'w+', encoding='UTF-8')
	# 对aishell进行文本数据处理
	for x in textobj.readlines():
		textlabel = x.strip('\n')
		textindex = textlabel.split('\t')[0]
		textlabel = textlabel.split('\t')[1]
		textlabel = textlabel.split(' ')
		x = pinyin(textlabel,style=Style.TONE3)
		str2 = ''
		for i in x:
			str1 = " ".join(i)
			if (re.search(r'\d',str1)):
				pass
			else:
				str1 += '5'
			str2 = str2 + ' ' + str1
		str2 = textindex + str2[:-1]
		# 保存生成的数据
		savefile.write(str2 + "\n")









def gen_source():
	f = open("set1_transcript.json", "r", encoding='utf-8')
	dict_list = json.load(f)
	print(len(dict_list))

	wavlist1 = open("train.wav.lst", "w", encoding='utf-8')
	syllabel1 = open("train.syllabel.txt", "w", encoding='utf-8')
	wavlist2 = open("test.wav.lst", "w", encoding='utf-8')
	syllabel2 = open("test.syllabel.txt", "w", encoding='utf-8')
	wavlist3 = open("dev.wav.lst", "w", encoding='utf-8')
	syllabel3 = open("dev.syllabel.txt", "w", encoding='utf-8')

	for x in dict_list:
		i = np.random.randint(10)
		index = x['id']
		text = x['text']
		file = x['file']
		mainfile = file[0]
		subfile = file[0:2]
		str1 = index + '\t' + text + '\n'
		str2 = index + '\t' + 'primewords_md_2018_set1\\audio_files\\' + mainfile + '\\' + subfile + '\\' + file + '\n'
		if i < 8:
			syllabel1.write(str1)
			wavlist1.write(str2)
		elif i == 8:
			syllabel2.write(str1)
			wavlist2.write(str2)
		else:
			syllabel3.write(str1)
			wavlist3.write(str2)

	syllabel1.close()
	wavlist1.close()
	syllabel2.close()
	wavlist2.close()
	syllabel3.close()
	wavlist3.close()


if __name__ == '__main__':
	gen_source()
	trans_aishell_to_pinyin('train.syllabel.txt', 'primewords\\train.syllabel.txt')
	trans_aishell_to_pinyin('test.syllabel.txt', 'primewords\\test.syllabel.txt')
	trans_aishell_to_pinyin('dev.syllabel.txt', 'primewords\\dev.syllabel.txt')