# Report
<center>王天乐

## Data Preprocessing

由于数据是tsv格式，且列之间采用Tab分割，所以利用python的csv库可以比较容易的进行读取:

`reader = csv.reader(open("test.tsv", "r", encoding="utf-8"), delimiter='\t')`

由于考虑到是文本匹配任务，所以文本次序并不重要，对于train data我们可以把label不变的情况下，交换一下左右的次序，这样可以把样本量翻倍，有助于训练。这样样本数目其实已经足够大了(70w+)，我尝试过进一步拓展增强数据，通过随机采样得到负样本，通过重复得到正样本，但测试发现相比于时间上的消耗这样带来的受益相当有限(提升<1%)。

## Model Introduction

我总共跑了两类模型，探究了目前SOTA的交互式匹配模型和特征式匹配模型的表现。

模型分别是交互式的BERT、RoBERTa和特征式的CoSENT，下面我将简单介绍一下这两类模型：

BERT[1] 是一个

RoBERTa[2]

CoSENT

之后我在查找paper的过程中，发现了两种交互式的方法，也就是Sentence-BERT和CoSENT，。

## My Implementation

这一节我就来讲讲我的模型性能是如何优化的吧。最开始我直接过一个BERT对两个语句分别求出768维的BERT representation，并直接将两者合并过一个linear层，得到最终结果，这一结果非常差劲，最终结果低于70%，现在想想这样也可以算是一种带弱交互的双塔模型。

之后我改用完全交互式的方法，使用BERT同时编码一个句子对，同样是直接过一个linear层，修改后结果提升到了73.451%。

然后我在此基础上进一步优化，首先增加了Data Preprocessing中数据增强的手段，同时在linear层之前增加了一个dropout层，进行这些改动后提升较为显著，表现达到了80.088%。

之后使用更强大的RoBERTa替换BERT，并选用更小的batch_size增加随机性，使用不断调整的learning_rate等手段，将表现提升到了86.725%。

而对于特征式匹配模型，我实现了表现比Sentence-BERT更优秀的CoSENT，不同于交互式模型，特征式模型通常是双塔结构，即句子对会返回两个各自的向量。而CoSENT正是力图调整BERT representation使得不同意的句子对之间的similarity尽量低，它通过特殊设计的loss来实现这一目标。最终测试时我们将搜寻一个最佳的threshold将这种similarity变为yes或no。我的实现配合上一步调好的RoBERTa，最终可以得到76.991%的结果。

实验代码：https://github.com/wtl666wtl/Text-Matching

## Performance & Analysis

各种实现的最终表现如下表所示：

|   Method    |  BERT   | BERT (with data augmentation) | RoBERTa (with more tricks) | RoBERTa + CoSENT |
| :---------: | :-----: | :---------------------------: | :------------------------: | :--------------: |
| Performance | 73.451% |            80.088%            |          86.725%           |     76.991%      |

可以看出RoBERTa相比于BERT确实优势比较明显，。同时，CoSENT和

## Conclusion



## Reference

[1] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

[2] Liu Y, Ott M, Goyal N, et al. Roberta: A robustly optimized bert pretraining approach[J]. arXiv preprint arXiv:1907.11692, 2019.