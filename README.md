# 深度学习中文语音识别系统

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

### 基于python的中文语音识别系统.
包含声学模型和语言模型两个部分组成，两个模型都是基于神经网络。

- 声学模型
   - 该项目实现了GRU-CTC中文语音识别声音模型，所有代码都在`gru_ctc_am.py`中，包括：
   - 增加了基于科大讯飞DFCNN的CNN-CTC结构的中文语音识别模型`cnn_ctc_am.py`，与GRU相比，对网络结构进行了稍加改造。
   
- 语言模型
   - 新增基于CBHG结构的语言模型`language_model\CBHG_lm.py`，该模型之前用于谷歌声音合成，移植到该项目中作为基于神经网络的语言模型。

- 数据集
   - 默认数据集为thchs30，参考gen_aishell_data中的数据及代码，也可以使用aishell的数据进行训练。
   - 增加将aishell数据处理为thchs30数据格式，合并数据进行训练。代码及数据放在`gen_aishell_data`中。
- 实验结果
   - 其中声学模型得到带有声调的拼音，如:
   ```python
   识别结果：jin1 zi1
   ```
   - 语言模型由拼音是别为汉字，如：
   ```python
   请输入测试拼音：ta1 mei2 you3 duo1 shao3 hao2 yan2 zhuang4 yu3 dan4 ta1 que4 ba3 ai4 qin1 ren2 ai4 jia1 ting2 ai4 zu3 guo2 ai4 jun1 dui4 wan2 mei3 de tong3 yi1 le qi3 lai2
   她没有多少豪言壮语但她却把爱亲人爱家庭爱祖国爱军队完美地统一了起来
   请输入测试拼音：chu2 cai2 zheng4 bo1 gei3 liang3 qian1 san1 bai3 wan4 yuan2 jiao4 yu4 zi1 jin1 wai4 hai2 bo1 chu1 zhuan1 kuan3 si4 qian1 wu3 bai3 qi1 shi2 wan4 yuan2 xin1 jian4 zhong1 xiao3 xue2
   除财政拨给两千三百万元教太资金外还拨出专款四千五百七十万元新建中小学
   请输入测试拼音：ke3 shi4 chang2 chang2 you3 ren2 gao4 su4 yao2 xian1 sheng1 shuo1 kan4 jian4 er4 xiao3 jie3 zai4 ka1 fei1 guan3 li3 he2 wang2 jun4 ye4 wo4 zhe shou3 yi1 zuo4 zuo4 shang4 ji3 ge4 zhong1 tou2
   可是常常有人告诉姚先生说看见二小姐在咖啡馆里和王俊业握着族一坐坐上几个钟头

   ```



该模型作为一个练手小项目。
增加语言模型，效果不错。

[我的github: https://github.com/audier](https://github.com/audier)

[我的博客: https://blog.csdn.net/chinatelecom08](https://blog.csdn.net/chinatelecom08)
