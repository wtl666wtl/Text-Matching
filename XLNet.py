import torch
from torch import nn
import transformers as tfs

model_class, tokenizer_class, pretrained_weights = (tfs.XLNetModel, tfs.XLNetTokenizer, 'xlnet-base-cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class XLNet(nn.Module):
    def __init__(self, k):
        super(XLNet, self).__init__()
        self.bert = model_class.from_pretrained(pretrained_weights)
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(768, k)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden, mems = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        linear_output = self.dense(torch.max(self.dropout(last_hidden), dim=1)[0])
        return linear_output, last_hidden
