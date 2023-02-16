import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import partial
from torch.nn import Parameter


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #init convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 2, 2)#3 input_layers, 16 output_layers, 3 kernel size, 2 step size
        self.conv2 = nn.Conv2d(16, 32, 2, 2)#16 input_layers, 32 output_layers, 3 kernel size, 2 step size
        self.conv3 = nn.Conv2d(32, 64, 2, 2)#32 input_layers, 64 output_layers, 3 kernel size, 2 step size
        self.conv4 = nn.Conv2d(64, 128, 2, 2)#64 input_layers, 256 output_layers, 3 kernel size, 2 step size

        #init batch normalization
        self.norm1 = nn.BatchNorm2d(16, affine=True)#affine= True means with learnable parameters
        self.norm2 = nn.BatchNorm2d(32, affine=True)#affine= True means with learnable parameters
        self.norm3 = nn.BatchNorm2d(64, affine=True)#affine= True means with learnable parameters
        self.norm4 = nn.BatchNorm2d(128, affine=True)#affine= True means with learnable parameters
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.downsample4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
    def forward(self, x):
        #Layer 1
        residual = x
        residual  = self.downsample1(residual)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = residual + x
        #print(x.size())
        
        #Layer2
        residual = x
        residual  = self.downsample2(residual)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = residual + x
        #print(x.size())

        #Layer3 
        residual = x
        residual  = self.downsample3(residual)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = residual + x
        #print(x.size())

        #Layer 4
        residual = x
        residual  = self.downsample4(residual)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = residual + x

        return x


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)

        #Reshape
        x = torch.reshape(x, (x.size(0), x.size(1), x.size(2) * x.size(3)))
        return x

def get_attention(queries, keys, values):
    scale = queries.shape[1] ** -0.5

    attention_scores = (queries @ keys.transpose(-2, -1)) * scale

    attention_probabilities = F.softmax(attention_scores, dim=-1)

    attention = attention_probabilities @ values

    return attention


class QueriesKeysValuesExtractor(nn.Module):
    def __init__(self, token_dim, head_dim, n_heads):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads

        queries_keys_values_dim = 3* self.head_dim * self.n_heads
        self.input_to_queries_keys_values = nn.Linear(
            in_features=token_dim, 
            out_features=queries_keys_values_dim, 
            bias=False
        )

    def forward(self, x):
        batch_size, n_tokens, token_dim = x.shape

        queries_keys_values = self.input_to_queries_keys_values(x)

        queries_keys_values = queries_keys_values.reshape(batch_size, 3, self.n_heads, n_tokens, self.head_dim)

        queries, keys, values = queries_keys_values.unbind(dim=1)

        return queries, keys, values

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, token_dim, head_dim, n_heads, dropout_p):
        super().__init__()
        self.qkv_extractor = QueriesKeysValuesExtractor(
            token_dim=token_dim,
            head_dim=head_dim,
            n_heads = n_heads
        )

        self.concatenated_heads_dim = n_heads * head_dim

        self.attention_to_output = nn.Linear(in_features=self.concatenated_heads_dim, out_features=token_dim)

        self.output_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, return_attention=False):
        batch_size, n_tokens, token_dim = x.shape

        queries, keys, values = self.qkv_extractor(x)

        attention = get_attention(queries=queries, keys=keys, values=values)
        attention_map = attention.clone()
        attention = attention.reshape(batch_size, n_tokens, self.concatenated_heads_dim)

        x = self.attention_to_output(attention)
        x = self.output_dropout(x)
        
        if return_attention == True:
            return x, attention_map
        else:
            return x


class TransformerBlock(nn.Module):
    def __init__(self, token_dim, mhsa_head_dim, mhsa_n_dim, multilayer_perceptron_dim, dropout_p, out_features):
        super().__init__()
        #init layer
        self.out_features=out_features

        self.layer_norm1 = nn.LayerNorm(normalized_shape=token_dim)

        self.multi_head_self_attention = MultiHeadSelfAttention(
            token_dim=token_dim,
            head_dim = mhsa_head_dim,
            n_heads = mhsa_n_dim,
            dropout_p = dropout_p
        )

        self.layer_norm2 = nn.LayerNorm(normalized_shape=token_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=token_dim, out_features=multilayer_perceptron_dim),
            nn.Dropout(p=dropout_p),
            nn.GELU(),
            nn.Linear(in_features=multilayer_perceptron_dim, out_features=out_features)
            #Maybe add extra dropout?
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.multi_head_self_attention(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        if self.out_features != 1:
            x = x + residual

        return x

class Encoder(nn.Module):
    def __init__(self, token_dim, mhsa_head_dim, mhsa_n_dim, multilayer_perceptron_dim, dropout_p):
        super().__init__()

        self.transformer_block1 = TransformerBlock(token_dim, mhsa_head_dim, mhsa_n_dim, multilayer_perceptron_dim, dropout_p, out_features=token_dim)

        self.transformer_block2 = TransformerBlock(token_dim, mhsa_head_dim, mhsa_n_dim, multilayer_perceptron_dim, dropout_p, out_features=1)

    def forward(self, x):
        x = self.transformer_block1(x)

        x = self.transformer_block2(x)

        return x
    
    def get_attention_maps(self, x):
        attention_maps = []
        
        _, attn_map1 = self.transformer_block1.multi_head_self_attention(x, return_attention=True)
        attention_maps.append(attn_map1)
        x = self.transformer_block1(x)
        
        _, attn_map2 = self.transformer_block2.multi_head_self_attention(x, return_attention=True)
        attention_maps.append(attn_map2)
        x = self.transformer_block2(x)
        
        return attention_maps

class MLPClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layerNorm = nn.LayerNorm(normalized_shape=in_features)

        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.layerNorm(x)

        x = self.linear(x)

        x = self.softmax(x)

        return x

class TokenConcatenator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        class_token = torch.zeros((batch_size, 128, 1))
        self.class_token = Parameter(class_token)

    def forward(self, x):

        class_token = self.class_token
        
        x = torch.cat((x, class_token), dim=-1)

        return x

class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        position_embedding = torch.zeros((1, 128, 197))
        self.position_embedding = Parameter(position_embedding)

    def forward(self, x):
        x = x + self.position_embedding
        return x

class CNN_T(nn.Module):
    def __init__(self, lr, batch_size, head_dim, mhsa_n_dim, multilayer_perceptron_dim, dropout_p):
        super(CNN_T, self).__init__()
        #init variables
        self.lr = lr
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.mhsa_n_dim = mhsa_n_dim
        self.multilayer_perceptron_dim = multilayer_perceptron_dim
        self.dropout_p = dropout_p
        
        #create the model
        #self._create_model()
        
    def create_model(self):
        self.convNet = ConvNet()

        self.patchEmbedding = PatchEmbedding()

        self.tokenConcatenator = TokenConcatenator(batch_size=self.batch_size)

        self.positionEmbedding = PositionEmbedding()

        self.encoder = Encoder(
            token_dim=197,#must stay the same
            mhsa_head_dim=self.head_dim,#can be changed
            mhsa_n_dim=self.mhsa_n_dim,#can be changed
            multilayer_perceptron_dim=self.multilayer_perceptron_dim,#can be changed
            dropout_p=self.dropout_p#can be changed
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((64, 1))

        self.mlp_classifier = MLPClassifier(64, 4)
        
    def forward(self, x):

        #convolutional model
        x = self.convNet(x)
        #print(x.size())

        #patch embedding
        x = self.patchEmbedding(x)
        #print(x.size())

        #concat with token
        x = self.tokenConcatenator(x)
        #print(x.size())

        #positional embedding
        x = self.positionEmbedding(x)
        #print(x.size())

        #transformer block
        x = self.encoder(x)
        #print(x.size())
        
        x = self.avgpool(x)
        
        #Reshape out_encoder
        x = torch.reshape(x, (x.size(0), x.size(1)))
        #print(x.size())
        
        
        
        #mlp
        x = self.mlp_classifier(x)
        #print(x.size())

        return x