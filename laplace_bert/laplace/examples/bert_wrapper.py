import torch
import numpy as np

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import (
BertForSequenceClassification
)

class BertMAP(nn.Module):

    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.config = bert_model.config
        self.num_labels = self.config.num_labels
    
    def forward(self, x, on_cuda=True):

        if isinstance(x, dict):
            x.pop('idx', None)
            if on_cuda:
                for k, v in x.items():
                    x[k] = v.cuda()
            output = self.bert_model(**x)
            return output.logits