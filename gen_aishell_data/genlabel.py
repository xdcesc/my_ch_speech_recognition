import os
import numpy as np
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
				

gen_label('dev.wav.lst', 'dev.syllable.txt')
