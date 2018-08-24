import os
import numpy as np
def gen_label(listfile,labelfile):
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
				

gen_label('test.wav.lst', 'test.syllabel.txt')
