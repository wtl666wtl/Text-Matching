import torch
from torch import nn
import transformers as tfs

model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizerFast, 'bert-large-cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class Bert_model(nn.Module):
    def __init__(self, k):
        super(Bert_model, self).__init__()
        self.bert = model_class.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(768, k)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        linear_output = self.dense(self.dropout(pooled))
        return linear_output, pooled
