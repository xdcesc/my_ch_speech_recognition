# -----------------------------------------------------------------------------------------------------
'''
&usage:		aishell数据处理，将汉字处理为拼音，并生成thchs30的数据形式
@author:	hongwen sun
'''
# -----------------------------------------------------------------------------------------------------
from pypinyin import pinyin, lazy_pinyin, Style
import numpy as np
import re

# -----------------------------------------------------------------------------------------------------
'''
usage: 将aishell汉字标注转化为拼音
env: pip install pypinyin
'''
# -----------------------------------------------------------------------------------------------------
def trans_aishell_to_pinyin(word_path, pinyin_path):
	# 需要转换为拼音的中文汉字路径
	textobj = open(word_path, 'r+', encoding='UTF-8')
	# 转化为拼音后的保存txt路径
	savefile = open(pinyin_path, 'w+', encoding='UTF-8')
	# 对aishell进行文本数据处理
	for x in textobj.readlines():
		textlabel = x.strip('\n')
		textlabel = textlabel.split(' ')
		x = pinyin(textlabel,style=Style.TONE3)
		str2 = ''
		for i in x:
			str1 = " ".join(i)
			if (re.search(r'\d',str1)):
				pass
			else:
				str1 += '5'
			str2 = str2 + str1 + ' '
		str2 = str2[:-1]
		# 保存生成的数据
		savefile.write(str2 + "\n")


# -----------------------------------------------------------------------------------------------------
'''
usage: 生成train, dev, test的音频文件列表
'''
# -----------------------------------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------------------------
'''
usage: 生成train, dev, test的音频文件对应的标注文件
'''
# -----------------------------------------------------------------------------------------------------
def gen_label(readfile,writefile):
	fileids = []
	content_dict = {}
	allfile = open('aishell_transcript.txt','r+', encoding='UTF-8')
	for textlabel in allfile.readlines():
		textlabel = textlabel.strip('\n')
		textlabel_id = textlabel.split(' ',1)[0]
		textlabel_text = textlabel.split(' ',1)[1]
		content_dict[textlabel_id] = textlabel_text
	listobj = open(readfile, 'r+', encoding='UTF-8')
	labelobj = open(writefile, 'w+', encoding='UTF-8')
	for content in listobj.readlines():
		label = ''
		content = content.strip('\n')
		content_id = content.split(' ',1)[0]
		if content_id in content_dict:
			content_text = content_dict[content_id]
			label = content_id + ' ' + content_text
			labelobj.write(label+'\n')
	labelobj.close()
	allfile.close()
	listobj.close()


# -----------------------------------------------------------------------------------------------------
'''
usage: 修正train, dev, test的音频文件列表，将标注中不存在的文件删除
'''
# -----------------------------------------------------------------------------------------------------
def fix_list(listfile,labelfile):
	fileids = []
	content_dict = {}
	allfile = open(listfile,'r+', encoding='UTF-8')
	for textlabel in allfile.readlines():
		textlabel = textlabel.strip('\n')
		textlabel_id = textlabel.split(' ',1)[0]
		textlabel_text = textlabel.split(' ',1)[1]
		content_dict[textlabel_id] = textlabel_text
	allfile.truncate()
	allfile.close()

	labelobj = open(labelfile, 'r+', encoding='UTF-8')
	listobj = open(listfile, 'w+', encoding='UTF-8')
	for content in labelobj.readlines():
		label = ''
		content = content.strip('\n')
		content_id = content.split(' ',1)[0]
		content_text = content_dict[content_id]
		label = content_id + ' ' + content_text
		listobj.write(label+'\n')
	labelobj.close()
	listobj.close()




# 将汉字标注化为拼音标注
# 在data_aishell同级目录下运行该脚本。
trans_aishell_to_pinyin('E:\\aishell_transcript_v0.8.txt', 'E:\\aishell_transcript1.txt')
# 生成train, dev, test的音频文件列表
gen_wavlist('data_aishell/wav/train','train.wav.lst')
gen_wavlist('data_aishell/wav/test','test.wav.lst')
gen_wavlist('data_aishell/wav/dev','dev.wav.lst')
# 生成train, dev, test的音频文件对应的标注文件
gen_label('train.wav.lst', 'train.syllable.txt')
gen_label('test.wav.lst', 'test.syllable.txt')
gen_label('dev.wav.lst', 'dev.syllable.txt')
# 修正train, dev, test的音频文件列表，将标注中不存在的文件删除
fix_list('train.wav.lst', 'train.syllable.txt')
fix_list('test.wav.lst', 'test.syllable.txt')
fix_list('dev.wav.lst', 'dev.syllable.txt')