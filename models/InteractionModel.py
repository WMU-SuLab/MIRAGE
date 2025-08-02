#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/20 14:57
# @Author : GaoShuaishuai
import numpy as np
import torch
import torch.nn as nn
class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))

    def forward(self, x):
        output = torch.mm(x, self.w) + self.bias
        return output

class SparseLinearLayer(nn.Module):
    def __init__(self, gene_size, device):
        super(SparseLinearLayer, self).__init__()
        self.device = device
        self.input_dim = sum(gene_size)
        self.output_dim = len(gene_size)
        self.mask = self._mask(gene_size).detach().to(self.device)

        scale = 1. * np.sqrt(6. / (self.input_dim + self.output_dim))
        # approximated posterior
        self.w = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale).to(self.device) * self.mask)
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale).to(self.device))

    def forward(self, x):
        output = torch.mm(x, self.w * self.mask) + self.bias
        return output

    def _mask(self, gene_size):
        index_gene = []
        index_gene.append(0)
        for i in range(len(gene_size)):
            index_gene.append(gene_size[i] + index_gene[i])
        sparse_mask = torch.zeros(sum(gene_size), len(gene_size))
        for i in range(len(gene_size)):
            sparse_mask[index_gene[i]:index_gene[i + 1], i] = 1
        return sparse_mask

class Gene_Block(nn.Module):
    def __init__(self, gene_size, device):
        super(Gene_Block, self).__init__()
        self.layer = SparseLinearLayer(gene_size, device)

    def forward(self, x):
        x = self.layer(x)
        return x


class Main_effect(nn.Module):
    def __init__(self, gene_size):
        super(Main_effect, self).__init__()
        self.input_dim = len(gene_size)
        self.Layer1 = LinearLayer(self.input_dim, 1)
        
    def forward(self, x):
        x = self.Layer1(x)
        return x



class InteractionNN(nn.Module):
    def __init__(self, gene_block, deepcvGRS):
        super(InteractionNN, self).__init__()
        self.gene_block = gene_block
        self.deepcvGRS = deepcvGRS

    def forward(self, x):
        x1 = self.gene_block(x)
        x2 = self.deepcvGRS(x1)
        return x2
        
        
        
class PermutationNN(nn.Module):
    def __init__(self, gene_block, main_effect):
        super(PermutationNN, self).__init__()
        self.gene_block = gene_block
        self.main_effect = main_effect

    def forward(self, x):
        x1 = self.gene_block(x)
        x2 = self.main_effect(x1)
        return x2