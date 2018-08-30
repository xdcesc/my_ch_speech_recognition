
"""
Before running this code, make sure that you've downloaded Leipzig Chinese Corpus 
(http://corpora2.informatik.uni-leipzig.de/downloads/zho_news_2007-2009_1M-text.tar.gz)
Extract and copy the `zho_news_2007-2009_1M-sentences.txt` to `data/` folder.

This code should generate a file which looks like this:
2[Tab]zhegeyemianxianzaiyijingzuofei...。[Tab]这__个_页_面___现___在__已_经___作__废__...。

In each line, the id, pinyin, and a chinese sentence are separated by a tab.
Note that _ menist@gmail.com
"""
from __future__ import print_function
import codecs
import os
import regex# pip install regex
from xpinyin import Pinyin # pip install xpinyin 

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
    
#读取fin中的中文数据，转化为拼音，按照一定格式将结果保存到fout中去
def build_corpus():
    with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/lable.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: break
                
                try:
                    # 对原文本处理，idx为句子的id，sent为句子的文本，保留汉字并求出句子拼音
                    line = line.strip('\n')
                    idx = line.split(' ', 1)[0]
                    sent = line.split(' ', 1)[1]
                    # 去除掉句子中的其他符号，保留汉字
                    sent = clean(sent)
                    if len(sent) > 0:
                        # 求出拼音，并对其
                        pnyns, hanzis = align(sent)
                        # 将结果保存到fout中
                        fout.write(u"{}\t{}\t{}\n".format(idx, pnyns, hanzis))
                except:
                    continue # it's okay as we have a pretty big corpus!
                
                if i % 10000 == 0: print(i, )
                i += 1

if __name__ == "__main__":
    build_corpus(); print("Done")
