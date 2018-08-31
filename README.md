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
   测试拼音：jin1 zi1
   金子
   ```



该模型作为一个练手小项目。
增加语言模型，效果不错。

[我的github: https://github.com/audier](https://github.com/audier)

[我的博客: https://blog.csdn.net/chinatelecom08](https://blog.csdn.net/chinatelecom08)
