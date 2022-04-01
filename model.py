import torch
from torch import nn
from torch import optim
import transformers as tfs
import math
import numpy as np

model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizerFast, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class Bert_model(nn.Module):
    def __init__(self, k):
        super(Bert_model, self).__init__()
        self.bert = model_class.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dense = nn.Linear(768 * 2, k)

    def forward(self, input_ids, attention_mask, input_ids2, attention_mask2):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask)
        _, pooled2 = self.bert(input_ids2, attention_mask=attention_mask2)
        tmp = torch.cat((pooled, pooled2), 1)
        linear_output = self.dense(tmp)
        return linear_output, tmp
