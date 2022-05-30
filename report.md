# Report
<center>王天乐

## Data Preprocessing

由于数据是tsv格式，且列之间采用Tab分割，所以利用python的csv库可以比较容易的进行读取:

`reader = csv.reader(open("test.tsv", "r", encoding="utf-8"), delimiter='\t')`

由于考虑到是文本匹配任务，所以文本次序并不重要，对于train data我们可以把label不变的情况下，交换一下左右的次序，这样可以把样本量翻倍，有助于训练。由于这样样本数目其实已经足够大了(70w+)，训练时间极长，就没有进一步使用其他数据增强的方法。

## Model Introduction

BERT[1] 是一个

RoBERTa[2]

## My Implement



## Performance & Analysis

这一节我就来讲讲我的模型性能是如何优化的吧。最开始我直接过一个BERT对两个语句分别求出768维的BERT representation，并直接将两者合并过一个linear层，得到最终结果，这一结果非常差劲，最终仅有69.469%。

之后我改用BERT同时编码一个句子对，同样是直接过一个linear层，修改后结果提升到了73.451%。

然后我在此基础上进一步优化，首先增加了Data Preprocessing中数据增强的手段，同时在linear层之前增加了一个dropout层，进行这些改动后提升较为显著，表现达到了80.088%。

之后使用RoBERTa替换BERT，并选用更小的batch_size增加随机性，使用不断调整的learning_rate等手段，将表现提升到了86.725%。

整体表现如下表所示：

|   Method    |  BERT   | BERT (with data augmentation) | RoBERTa (with more tricks) | CoSent |
| :---------: | :-----: | :---------------------------: | :------------------------: | :----: |
| Performance | 73.451% |            80.088%            |          86.725%           |        |



## Conclusion



## Reference

[1] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

[2] Liu Y, Ott M, Goyal N, et al. Roberta: A robustly optimized bert pretraining approach[J]. arXiv preprint arXiv:1907.11692, 2019.