from xpinyin import Pinyin # pip install xpinyin 
import regex
import codecs
from collections import Counter
from itertools import chain
import re
import tensorflow as tf

def align(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A tuple of pinyin and chinese sentence.
    '''
    pinyin = Pinyin()
    # 求出汉字拼音，并且放到list中去
    pnyns = pinyin.get_pinyin(sent, " ").split()
    # 在漢字中添加_
    hanzis = []
    for char, p in zip(sent.replace(" ", ""), pnyns):
        hanzis.extend([char] + ["_"] * (len(p) - 1))
        
    pnyns = "".join(pnyns)
    hanzis = "".join(hanzis)
    
    assert len(pnyns) == len(hanzis), "The hanzis and the pinyins must be the same in length."
    return pnyns, hanzis



def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text

if __name__ == '__main__':
	cout = []
	cout.append('1112315456')
	cout.append('42535')
	print(cout)
