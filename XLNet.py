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
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(768, k)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_tokens = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        cls_tokens = output_tokens[:, 0, :]
        linear_output = self.dense(self.dropout(cls_tokens))
        return linear_output, cls_tokens
