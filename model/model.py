import torch.nn as nn
from transformers import RobertaModel

# Model with extra layers on top of RoBERTa
class ROBERTAClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = nn.Dropout(dropout_rate)
        self.l1 = nn.Linear(768, 64)
        self.bn1 = nn.LayerNorm(64)
        self.d2 = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        
        return x  