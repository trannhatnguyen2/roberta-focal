import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class PhoBertFeedForward_base(nn.Module):
    def __init__(self, from_pretrained:bool=True, freeze_backbone:bool=False, drop_out:float=0.1, out_channels:int=2):
        super(PhoBertFeedForward_base, self).__init__()
        phobert_config = RobertaConfig.from_pretrained("vinai/phobert-base-v2")
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained("vinai/phobert-base-v2")
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(drop_out),
            nn.Linear(768, out_channels))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits