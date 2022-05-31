import torch
from torch import nn
import transformers as tfs

model_class, tokenizer_class, pretrained_weights = (tfs.RobertaModel, tfs.RobertaTokenizer, 'roberta-base')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class CoSent(nn.Module):
    def __init__(self):
        super(CoSent, self).__init__()
        self.bert = model_class.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return pooled