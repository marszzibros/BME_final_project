import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

class ResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze_half=True, num_classes=4):
        super(ResNet, self).__init__()

        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None
        )

        for param in self.resnet.parameters():
            param.requires_grad = True

        if freeze_half:
            num_layers = len(list(self.resnet.children()))
            frozen_layers = num_layers // 2
            for i, child in enumerate(self.resnet.children()):
                if i < frozen_layers and i != 0:
                    for param in child.parameters():
                        param.requires_grad = False

        # remove classification heads
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 

        self.fc = nn.Linear(in_features, num_classes) 

    def forward(self, x):
        features = self.resnet(x) 
        x = self.fc(features) 

        # multi class classifications
        return F.log_softmax(x, dim=1)
    
class ResNet_less_train(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze_half=True, num_classes=4):
        super(ResNet_less_train, self).__init__()

        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None
        )

        for param in self.resnet.parameters():
            param.requires_grad = True

        if freeze_half:
            num_layers = len(list(self.resnet.children()))
            frozen_layers = 3 * num_layers // 4
            for i, child in enumerate(self.resnet.children()):
                if i < frozen_layers and i != 0:
                    for param in child.parameters():
                        param.requires_grad = False

        # remove classification heads
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 

        self.fc = nn.Linear(in_features, num_classes) 

    def forward(self, x):
        features = self.resnet(x) 
        x = self.fc(features) 

        # multi class classifications
        return F.log_softmax(x, dim=1)
    
class ViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze=True):
        super(ViT, self).__init__()

        self.vit = models.vit_b_16(pretrained=pretrained)

        self.vit.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.vit.conv_proj.out_channels,
            kernel_size=self.vit.conv_proj.kernel_size,
            stride=self.vit.conv_proj.stride,
            padding=self.vit.conv_proj.padding,
            bias=self.vit.conv_proj.bias is not None
        )

        for param in self.vit.parameters():
            param.requires_grad = True

        if freeze:
            total_layers = len(self.vit.encoder.layers)
            for layer_idx in range(0, total_layers - 4):
                for param in self.vit.encoder.layers[layer_idx].parameters():
                    param.requires_grad = False

        # Replace the classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return F.log_softmax(self.vit(x), dim = 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[num_tokens, batch_size, embedding_dim]``
        """

        seq_len = x.size(0)
        return self.dropout(x + self.pe[:seq_len])

class Transformer(nn.Module):
    def __init__ (self, 
                  ntoken: int, 
                  d_model: int, 
                  nhead: int, 
                  d_hid:int,
                  nlayers: int,
                  seq_len: int,
                  dropout: float = 0.5):
        super(Transformer, self).__init__()

        self.ntoken = ntoken
        self.d_model = d_model
        self.seq_len = seq_len

        # Embedding
        self.embedding = nn.Linear(seq_len * seq_len, d_model * seq_len)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len = self.seq_len)

        # Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(d_model, 4)

    def forward(self, x):

        BS, C, H, W  = x.shape
        x = x.reshape(BS,H * W)
        
        
        x = self.embedding(x) * math.sqrt(self.d_model)

        x = self.pos_encoder(x.reshape(H, BS, self.d_model))

        # add classification token (cls token) in the sequence 
        # x   : (40, BS, 32)
        CLS_token = torch.zeros(1, BS, self.d_model)
        x = torch.cat([CLS_token.to(x.device), x], axis=0).permute(1,0,2)

        # Transformer encoder (num_tokens, BS, embed_dim)
        encoder_output = self.transformer_encoder(x)[:,0,:]
        encoder_output = encoder_output.view(encoder_output.shape[0], encoder_output.shape[1])
        
        return F.log_softmax(self.fc(encoder_output), dim = 1)
