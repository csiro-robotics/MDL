"""
Author: Zhibin Li: lzb5600@gmail.com
"""
import copy
import torch
import torch.nn.functional as F
import numpy as np


                
class corr_meta_learning(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, inner_step=1, inner_step_size=1e-3, lmbd=1, layer=3):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)      
        self.inner_step = inner_step
        self.inner_step_size = inner_step_size
        self.layer = layer
        self.lmbd = lmbd
        self.projection = sparse_proj(self.num_fields, lmbd)
        self.projection_w = proj_w()
        self.feature_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.w = torch.nn.Parameter(torch.empty(self.num_fields, self.num_fields))
        torch.nn.init.constant_(self.w.data, 1.0/(self.num_fields-1))
        self.w.data.fill_diagonal_(-1)
        self.register_buffer('mu', torch.full((self.num_fields,), lmbd/self.num_fields) )
        #==========backbone========================       
        row, col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i), col.append(j)     
        self.cross_idx = [row, col]
        self.kernel = torch.nn.Parameter(torch.empty(len(row), self.embed_dim, self.embed_dim)) 
        torch.nn.init.xavier_uniform_(self.kernel.data)
        in_dim = int(self.num_fields*(self.num_fields-1)/2 + self.num_fields**2 - self.num_fields)
        layers = list()
        for n_layer in range(self.layer-1):
            layers.append(torch.nn.BatchNorm1d(in_dim))
            layers.append(torch.nn.Linear(in_dim, 100))
            layers.append(torch.nn.ReLU())
            in_dim = 100
        layers.append(torch.nn.Linear(in_dim, 1, bias=True))
        self.clf = torch.nn.Sequential(*layers)
        self.linear = FeaturesEmbedding(field_dims,1)
        #==========================================


    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """ 
        vx = self.feature_embedding(x)
        w = self.w.expand(vx.shape[0],*self.w.shape)
        mu = self.mu.expand(vx.shape[0],*self.mu.shape)
        wv = w.bmm(vx)
        #=========inner loop of MDL===========
        for step in range(self.inner_step):
            dw = wv.bmm(vx.transpose(1,2))
            dw = (mu/(self.lmbd)).unsqueeze(2).mul(dw)
            w = w - self.inner_step_size * dw
            w = self.projection_w(w)
            wv = w.bmm(vx)
            dmu = 0.5 * (wv**2).sum(2)
            mu = mu - self.inner_step_size * dmu
            mu = self.projection(mu)
        #=====================================
        ox = vx[:,self.cross_idx[0],:].unsqueeze(2).matmul(self.kernel).squeeze(2)
        ox = torch.einsum('ijk,ijk->ij', ox, vx[:,self.cross_idx[1],:])
        w = mu.unsqueeze(2).mul(w)
        w = w.view(-1,w.shape[1]**2)[:,1:].view(-1,w.shape[1]-1,w.shape[1]+1)[:,:,:-1] #extract off-diagonal elements
        ox = torch.cat( (w.reshape(w.shape[0],-1),ox), 1 )
        ox = self.clf(ox).squeeze(1)
        ox = ox + self.linear(x).squeeze().sum(-1)
        return ox
        
        

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.register_buffer("offsets", torch.tensor((0,*field_dims.cumsum(0)[:-1].astype(np.long)) ).unsqueeze(0) )
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + self.offsets
        return self.embedding(x)
        
        
        
class sparse_proj(torch.nn.Module):

    def __init__(self, num_fields,  lmbd=1):
        super().__init__()
        self.lmbd = lmbd
        self.num_fields = num_fields
        self.register_buffer('id_range', torch.arange(1,num_fields+1) )

    def forward(self,z): 
        z_s = torch.sort(z,dim=1,descending=True).values
        z = z - z_s[:,[0]]
        z_s = z_s - z_s[:,[0]]
        z_s_cumsum = torch.cumsum(z_s, dim=1)
        
        taus = (z_s_cumsum - self.lmbd) / self.id_range
        k =  (z_s > taus).sum(1,keepdim=True)
        tau = torch.gather(taus, dim=1, index=k-1)
        p_z  = F.relu(  z - tau )
        return p_z


class proj_w(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,z):
        z.diagonal(dim1=1, dim2=2)[:] = -1
        return z

