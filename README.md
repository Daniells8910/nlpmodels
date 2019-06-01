## 工程结构：<br>
configs：yaml格式的配置文件，所有训练参数均存放进该文件<br>
datasets:常用的数据集<br>
[tab]* dataset.py：生成训练数据集、测试数据集、词汇表；支持读取预训练向量文件<br>
models：模型实例化，每个模型文件包括ModelParam和ModelTrain两个主要类，ModelTrain类中可体现单模型以及多模型的拼装操作<br>
tasks：各种NLP任务<br>
[tab]* classification：分类任务，支持机器学习模型和深度学习模型<br>
[tab]* ner：命名实体识别任务<br>
[tab]* seq2seq：<br>
tools：各种文件处理工具，包括文件预处理、分词工具等<br>
[tab]* regcommon.py：基于正则表达式对文本进行处理<br>
[tab]* segment.py：基于jieba创建Segment文件<br>
[tab]* langconv,py：更改繁体为简体<br>
examples:样例，包括提供样例文件<br>
