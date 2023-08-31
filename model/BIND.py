import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.vals = []
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.vals = []
        self.val = None
        self.avg = 0

    def update(self, val):
        self.vals.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

        
class SoftPlus(nn.Module):

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-8, dim=None):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        # apply softplus to first dim dimension
        if self.dim is None:
            result = self.Softplus(x) + self.epsilon
        else:
            result = torch.cat((self.Softplus(x[..., :self.dim])+self.epsilon, x[..., self.dim:]), dim=-1)

        return result
    

class MLP(nn.Module): 
    
    def __init__(self, dim_in, dim_out, dim_hidden = 16, num_hidden = 0, activation = nn.ReLU()):
        super(MLP, self).__init__()
        
        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden - 1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')
        self.dropout = nn.Dropout(0.2)
        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)
            
        self.activation = activation
        
    def forward(self, x):
        for m in self.linears[:-1]:
            x = m(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.dropout(x)
        return self.linears[-1](x)#最后来一波linear直上nan
    
class RNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, activation):
        super(RNN, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.i2h = MLP(dim_in + dim_hidden, dim_hidden, dim_hidden, num_hidden, activation)
        self.h2o = MLP(dim_hidden, dim_out, dim_hidden, num_hidden, activation)
        self.activation = activation
        
    def forward(self, x, h0 = None):
        assert len(x.shape) > 2,  'z need to be at least a 2 dimensional vector accessed by [tid ... dim_id]'
        
        if h0 is None:
            h0 = [torch.zeros(x.shape[1] + (self.dim_hidden,))]
        else:
            hh = [h0]
            
        for i in range(x.shape[0]):
            combined = torch.cat((x[i], hh[-1]), dim=-1)
            hh.append(self.activation(self.   i2h(combined)))
            
        return self.h2o(torch.stack(tuple(hh)))
        
        
class Adaptive_Adj(nn.Module):
    def __init__(self, batch_num, Node_num, sequence_length, dim_in, dim_mid, device, saturation = 1, saturation_= 1):
        super(Adaptive_Adj, self).__init__()
        
        self.batch_size = batch_num
        self.Node_num = Node_num
        self.sequence_length = sequence_length
        self.saturation = saturation
        self.saturation_ = saturation_
        self.dim_in = dim_in
        self.dim_mid = dim_mid

        self.k = int((Node_num * sequence_length) // 2)
        self.sm = nn.Softmax(dim = -1)
        self.device = device
        
        ####################
        
        self.embs1 = nn.Embedding(Node_num, sequence_length)
        self.embs2 = nn.Embedding(Node_num, sequence_length)
        self.embt1 = nn.Embedding(sequence_length, Node_num)
        self.embt2 = nn.Embedding(sequence_length, Node_num)
        
        self.idx_s = torch.arange(Node_num).to(device)
        self.idx_t = torch.arange(sequence_length).to(device)
        self.Is = nn.Parameter(torch.ones(Node_num).to(device))
        self.It = nn.Parameter(torch.ones(sequence_length).to(device))

        self.lint1 = nn.Linear(dim_in, dim_mid)

        self.lins1 = nn.Linear(dim_in, dim_mid)

        self.ws_ = nn.Parameter(torch.eye(dim_mid).to(device))
        self.wt_ = nn.Parameter(torch.eye(dim_mid).to(device))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.maxpool_t1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool_s1 = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):#BTNF
        
        embs1 = self.embs1(self.idx_s)#N,dim_mid
        embs2 = self.embs2(self.idx_s)
        
        embt1 = self.embt1(self.idx_t)#T, dim_mid
        embt2 = self.embt2(self.idx_t)

        adjS = torch.matmul(embs1, embs2.T) - torch.matmul(embs2, embs1.T)
        adjS = F.relu(self.saturation * adjS)
        

        adjT = torch.matmul(embt1, embt2.T) - torch.matmul(embt2, embt1.T)
        adjT = F.relu(self.saturation * adjT)
        
        
        ####################
        ####################
        
       
        x_pooling = self.avgpool(x.reshape(-1, self.sequence_length*self.Node_num*self.dim_in).transpose(0,1))
        x_pooling = x_pooling.reshape(self.sequence_length,self.Node_num,self.dim_in)
        
        
        x_t1 = self.lint1(x_pooling).transpose(1,2)
        x_t1 = self.maxpool_t1(x_t1).squeeze(-1)
        
        x_s1 = self.lins1(x_pooling).permute(1,2,0)
        x_s1 = self.maxpool_s1(x_s1).squeeze(-1)

        x_t = torch.matmul(x_t1,self.wt_)
        x_t = torch.matmul(x_t, x_t1.transpose(0,1))
        x_s = torch.matmul(x_s1,self.ws_)
        x_s = torch.matmul(x_s, x_s1.transpose(0,1))
        x_t = F.relu(self.saturation_*x_t)
        x_s = F.relu(self.saturation_*x_s)
        
#         adjT = torch.eye(self.sequence_length).to(device)
#         x_t = torch.eye(self.sequence_length).to(device)
        
#         #####################
#         #####################

        At_As = torch.kron(adjT, adjS)
        At_As_ = self.sm(torch.kron(x_t, x_s))
        
        return At_As_*At_As



#将每一个adj的卷积连续化
class odeFunc(nn.Module):
    def __init__(self,dim_in, dim_out, dim_mid, dim_N, device, ablation = None, cheby_k = 4, Cheby = True):
        super(odeFunc, self).__init__()
        self.adj = None
        self.cheby_k = cheby_k
        self.dim_N = dim_N
        self.dim_in = dim_in
        self.device = device
        
        self.in_to_mid = nn.Linear(dim_in, dim_mid)
        add_random = True
        self.w_trans = nn.Parameter(torch.eye(dim_mid).to(device)) if add_random else torch.eye(dim_mid).to(device)
        self.sm = nn.Softmax(-1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Cheby = Cheby
        self.ablation = ablation
        print(ablation,'##')
        if (ablation == 'woode') or (ablation is None):
            in_cha =  2 * cheby_k * dim_in if Cheby else (1 + 2 * cheby_k) * dim_in
        else:
            in_cha =  1 * cheby_k * dim_in if Cheby else (1 + 2 * cheby_k) * dim_in
        mid_cha = in_cha // 2
        self.lin = nn.Sequential(nn.Conv1d(in_cha, mid_cha,1,1,bias = True), nn.ReLU(),
                                 nn.Conv1d(mid_cha, dim_mid,1,1,bias = True),nn.ReLU(),
                                 nn.Conv1d(dim_mid, dim_out,1,1,bias = True))
        self.out_adj = None
        
    def set_init_adj(self,adj):
        self.adj = adj
    def set_outadj(self,out_adj):
        self.out_adj = out_adj
        
    def forward(self,t,x):
        if self.ablation == 'wokagg':
            ###############
            #attention
            print(x.shape,'##')
            x_adjx = self.avgpool(x.permute(1,2,0)).squeeze(-1)
            x_adjx = self.in_to_mid(x_adjx.transpose(0,1))
            adjx = torch.matmul(x_adjx, self.w_trans)#nc * cc -> nc
            adjx = torch.matmul(adjx, x_adjx.transpose(0,1))#nc * cn -> nn
            adjx = self.sm(F.relu(adjx))
            self.set_outadj(adjx)
            adjxs = [torch.eye(self.dim_N).to(self.device), adjx]
            for k in range(2, self.cheby_k): 
                adjxs.append(torch.matmul(2 * adjx, adjxs[-1]) - adjxs[-2])
            adjxs = torch.stack(adjxs, dim=0)#knn
            x_adjx = torch.einsum('bcn,knm->bckm',x,adjxs)
            x_adjx = x_adjx.reshape(-1,self.cheby_k*self.dim_in,self.dim_N)
            x = self.lin(x_adjx)
        elif self.ablation == 'woatten':
            if self.Cheby:
                adjs = [torch.eye(self.dim_N).to(self.device), self.adj]
                for k in range(2, self.cheby_k):
                    adjs.append(torch.matmul(2 * self.adj, adjs[-1]) - adjs[-2])
                adjs = torch.stack(adjs, dim=0)#knn
                x_adj = torch.einsum('bcn,knm->bckm',x,adjs)
                x_adj = x_adj.reshape(-1,self.cheby_k*self.dim_in,self.dim_N)
            else:
                x_adj_0 = x
                x_adj_1 = torch.einsum('bcn,nm->bcm',x,self.adj)
                x_adj = torch.cat([x, x_adj_1],1)
                for k in range(2, self.cheby_k + 1):
                    x_adj_2 = 2 * torch.einsum('bcn,nm->bcm',x_adj_1,self.adj) - x_adj_0
                    x_adj = torch.cat([x_adj, x_adj_2],1)
                    x_adj_1, x_adj_0 = x_adj_2, x_adj_1
            x = self.lin(x_adj)
        else:
            if self.Cheby:
                adjs = [torch.eye(self.dim_N).to(self.device), self.adj]
                for k in range(2, self.cheby_k):
                    adjs.append(torch.matmul(2 * self.adj, adjs[-1]) - adjs[-2])
                adjs = torch.stack(adjs, dim=0)#knn
                x_adj = torch.einsum('bcn,knm->bckm',x,adjs)
                
                x_adj = x_adj.reshape(-1,self.cheby_k*self.dim_in,self.dim_N)
            else:
                x_adj_0 = x
                x_adj_1 = torch.einsum('bcn,nm->bcm',x,self.adj)
                x_adj = torch.cat([x, x_adj_1],1)
                for k in range(2, self.cheby_k + 1):
                    x_adj_2 = 2 * torch.einsum('bcn,nm->bcm',x_adj_1,self.adj) - x_adj_0
                    x_adj = torch.cat([x_adj, x_adj_2],1)
                    x_adj_1, x_adj_0 = x_adj_2, x_adj_1

            ###############
            #attention
            x_adjx = self.avgpool(x.permute(1,2,0)).squeeze(-1)
            x_adjx = self.in_to_mid(x_adjx.transpose(0,1))
            adjx = torch.matmul(x_adjx, self.w_trans)#nc * cc -> nc
            adjx = torch.matmul(adjx, x_adjx.transpose(0,1))#nc * cn -> nn
            adjx = self.sm(F.relu(adjx))
            self.set_outadj(adjx)
            adjxs = [torch.eye(self.dim_N).to(self.device), adjx]
            for k in range(2, self.cheby_k): 
                adjxs.append(torch.matmul(2 * adjx, adjxs[-1]) - adjxs[-2])
            adjxs = torch.stack(adjxs, dim=0)#knn
            x_adjx = torch.einsum('bcn,knm->bckm',x,adjxs)
            x_adjx = x_adjx.reshape(-1,self.cheby_k*self.dim_in,self.dim_N)

            x = torch.cat([x_adj,x_adjx],dim=1)

            x = self.lin(x)
            #b,dim_out,N
        return x

    
class Mixprop(nn.Module):
    def __init__(self,dim_in, dim_out, dim_mid ,dim_N, device, atol = 1e-5, rtol = 1e-5, step_num = 2, ablation = None):#try 3
        super(Mixprop, self).__init__()
        self.res = nn.Sequential(nn.Conv1d(dim_in, dim_out, 1, 1))#,nn.BatchNorm1d(dim_out))
        if ablation == 'woode':
            self.wrap = nn.Sequential(nn.Conv1d((1 + 1 ) * dim_out, dim_out, 1, 1))
        else:
            self.wrap = nn.Sequential(nn.Conv1d((1 +  1 * (step_num - 1)) * dim_out, dim_out, 1, 1))#,nn.BatchNorm1d(dim_out))
        self.dim_N = dim_N
        self.atol = atol
        self.rtol = rtol
        self.device = device
        self.GNNODE = odeFunc(dim_out, dim_out,dim_mid, dim_N, device,ablation)
        self.step_num = step_num
        self.t = torch.linspace(0,1, steps = step_num).to(device)
        self.drop = nn.Dropout(0)
        self.ablation = ablation
    def forward(self,x,adj):
        #B N dim_out 
        x = x.transpose(1,2)#B,dim_in + dim_out,N
        residue = self.res(x)#B,dim_out,N
        ho_0 = residue
        
        self.GNNODE.set_init_adj(adj)
        
        if self.ablation == 'woode':
            ho = self.GNNODE(0,ho_0)
            assert len(ho.shape) == 3, f'ho shape:{ho.shape}'
        else:
            ho = odeint(self.GNNODE, ho_0, self.t, method = 'euler',rtol = self.rtol, atol = self.atol)[1:]
            T,B,F,N = ho.shape
            ho = ho.permute(1,0,2,3).reshape(B, T*F, N)      
        outadj = self.GNNODE.out_adj
        
        h_O = self.wrap(self.drop(torch.cat([ho,residue],dim=1)))
        if torch.any(torch.isnan(h_O)):
            print('BOOOOOM, NaN in Mixprop!')
        if outadj is not None:
            return h_O,outadj.detach()#b,dim_out,N
        else:
            return h_O,_
        

class Graph_sampler(nn.Module):
    def __init__(self, dim_in, dim_out, batch_num, Node_num, sequence_length, device, 
                 out_sequence_length = 12, dim_mid = 8, dim_output = 1, 
                 atol = 1e-5, rtol = 1e-5, 
                 WEIGHTED = False, time_divides = [3], time_divides_ = [4], 
                 L_set = [3,7], S_set = [2,3], L_reso = None, S_reso = None, Ablation = False):
        super(Graph_sampler, self).__init__()
        
        self.start_lin = nn.Linear(dim_in, dim_out)
        Layer_num = []
        Left_dim = []
        
        Kernel_set = [L_set,S_set]
        Reso = [L_reso,S_reso]
        Layer_num = []
        Left_dim = []
        for hl in range(2):
            Length = sequence_length - Reso[hl] + 1
            lis =  Kernel_set[hl]
            len_sum = np.sum(lis)
            ke_num = len(lis)
            assert len_sum <= Length - 1
            i = -1
            length = Length
            while length >= 1:
                length = length - (len_sum - ke_num)
                i += 1
            Layer_num.append(i)
            Left_dim.append((length + (len_sum - ke_num)))
        
        self.L_reso = L_reso
        self.S_reso = S_reso
        if L_set is not None:
            assert (len(L_set) == len(S_set)) and (len(L_set) in (1,2)) ,'len L_set: {len(L_set)}; len S_set: {len(S_set)}'
        if len(L_set) == 2:
            self.L_extractor = nn.Sequential(*([nn.Conv2d(dim_out, dim_out, (1,L_set[0]), 1), nn.ReLU(),
                                                nn.Conv2d(dim_out, dim_out, (1,L_set[1]), 1), ]*Layer_num[0] 
                                               + [nn.Conv2d(dim_out,dim_out,(1,Left_dim[0]), 1),nn.Linear(1,L_reso if L_reso is not None else sequence_length)]))
            self.S_extractor = nn.Sequential(*([nn.Conv2d(dim_out, dim_out, (1,S_set[0]), 1), nn.ReLU(),
                                                nn.Conv2d(dim_out, dim_out, (1,S_set[1]), 1), ]*Layer_num[1] 
                                               + [nn.Conv2d(dim_out,dim_out,(1,Left_dim[1]), 1), nn.Linear(1,S_reso if S_reso is not None else sequence_length)]))
        elif len(L_set) == 1:
            self.L_extractor = nn.Sequential(*([nn.Conv2d(dim_out, dim_out, (1,L_set[0]), 1), nn.ReLU(),]*Layer_num[0] 
                                               + [nn.Conv2d(dim_out,dim_out,(1,Left_dim[0]), 1)]))#nn.Linear(1,L_reso if L_reso is not None else sequence_length)]))
            self.S_extractor = nn.Sequential(*([nn.Conv2d(dim_out, dim_out, (1,S_set[0]), 1), nn.ReLU(), ]*Layer_num[1] 
                                               + [nn.Conv2d(dim_out,dim_out,(1,Left_dim[1]), 1)])) #nn.Linear(1,S_reso if S_reso is not None else sequence_length)]))
            
           
        time_divides = [1] if L_reso is not None else time_divides
        time_divides_ = [1] if S_reso is not None else time_divides_
        self.time_divides = time_divides
        self.time_divides_ = time_divides_
        self.sequence_length = sequence_length
        self.Node_num = Node_num
        chunks = [(L_reso if L_reso is not None else self.sequence_length) // i for i in time_divides]
        chunks_ = [(S_reso if S_reso is not None else self.sequence_length) // i for i in time_divides_]
        
        self.chunks = chunks
        self.chunks_ = chunks_
        
        self.dim_Ns = [Node_num * k for k in chunks]
        self.dim_Ns_ = [Node_num * k for k in chunks_]
        
        self.out_sequence_length = out_sequence_length
        self.dim_out = dim_out
        self.batch_num = batch_num
        self.dim_output = dim_output
       
        self.adaptive_adj = nn.ModuleList()
        self.mixprop = nn.ModuleList()
        self.adaptive_adj_ = nn.ModuleList()
        self.mixprop_ = nn.ModuleList()
        # Long
        for i,(chunk,dim_N) in enumerate(zip(self.chunks,self.dim_Ns)):
            self.adaptive_adj.append(Adaptive_Adj(batch_num, Node_num, chunk, dim_out, dim_mid, device))
            self.mixprop.append(Mixprop(dim_out, dim_out, dim_mid ,dim_N, device, atol = atol, rtol = rtol, step_num = 3, ablation = Ablation))
            
        layer = len(self.time_divides)
        self.layer = layer
        self.Nconv = nn.Sequential(*([nn.Conv2d(L_reso if L_reso is not None else sequence_length, dim_mid, (1,dim_out),(1,1)),nn.ReLU()] +
                                     [nn.Conv2d(dim_mid, dim_mid, (1,dim_out),(1,1)), nn.ReLU()] * (layer - 1) + 
                                     [nn.Conv2d(dim_mid, out_sequence_length,(1,layer), (1,1))]))
        
        # Short
        for i,(chunk,dim_N) in enumerate(zip(self.chunks_,self.dim_Ns_)):
            self.adaptive_adj_.append(Adaptive_Adj(batch_num, Node_num, chunk, dim_out , dim_mid, device))
            self.mixprop_.append(Mixprop(dim_out, dim_out, dim_mid ,dim_N, device, atol = atol, rtol = rtol, step_num = 3, ablation = Ablation))
        
        layer_ = len(self.time_divides_)
        self.layer_ = layer_
        self.Nconv_ = nn.Sequential(*([nn.Conv2d(S_reso if S_reso is not None else sequence_length, dim_mid, (1,dim_out),(1,1)),nn.ReLU()] +
                                     [nn.Conv2d(dim_mid, dim_mid, (1,dim_out),(1,1)), nn.ReLU()] * (layer_ - 1) + 
                                     [nn.Conv2d(dim_mid, out_sequence_length,(1,layer_), (1,1))]))
        self.gate_conv = nn.Conv2d(2 * self.out_sequence_length, self.out_sequence_length,1,1)
        self.WEIGHTED = WEIGHTED
        self.device = device
        
        ###############
        
    def forward(self, x):#BTNC输入
        x = self.start_lin(x[0])

        x = x.permute(0,3,2,1)#BCNT

        b,c,n,t = x.shape
        x_long = self.L_extractor(x).reshape(b,self.dim_out,n,self.L_reso if self.L_reso is not None else self.sequence_length).permute(0,3,2,1)
        x_short = self.S_extractor(x).reshape(b,self.dim_out,n,self.S_reso if self.S_reso is not None else self.sequence_length).permute(0,3,2,1)#BTNC
        signals_pools = []
        for ind,time_divide in enumerate(self.time_divides):
            signals_pool = []
            for i in range(time_divide):
                x_tar = x_long[:,i*self.chunks[ind]:(i+1)*self.chunks[ind],:,:] 
                subgraph_set = self.adaptive_adj[ind](x_tar) #TN,TN
                x_tar = x_tar.reshape(-1, self.dim_Ns[ind], self.dim_out)#在val时不需要padding，都是通过前几时间片预测后几时间片
                signals,outadj = self.mixprop[ind](x_tar, subgraph_set)
                signals_pool.append(signals)
            signals_pool = torch.stack(signals_pool, 0).transpose(2,3)
            gn_dot_td, bs, gs, do = signals_pool.shape
            assert do == self.dim_out
            signals_pool = signals_pool.permute(1,0,2,3).reshape(bs,gn_dot_td * gs, do)#64,2040,32
            signals_pool = signals_pool.reshape(bs,self.L_reso if self.L_reso is not None else self.sequence_length,self.Node_num,do)
            signals_pools.append(signals_pool)
        signals_pools = torch.cat(signals_pools, dim = -1)
        
        assert self.layer * self.dim_out == signals_pools.shape[-1]
        out_long = self.Nconv(signals_pools)
        
        signals_pools = []
        for ind,time_divide in enumerate(self.time_divides_):
            signals_pool = []
            for i in range(time_divide):
                x_tar = x_short[:,i*self.chunks_[ind]:(i+1)*self.chunks_[ind],:,:] 
                subgraph_set = self.adaptive_adj_[ind](x_tar) #TN,TN
                x_tar = x_tar.reshape(-1, self.dim_Ns_[ind], self.dim_out)#在val时不需要padding，都是通过前几时间片预测后几时间
                signals,outadj_ = self.mixprop_[ind](x_tar, subgraph_set)
                signals_pool.append(signals)
            signals_pool = torch.stack(signals_pool, 0).transpose(2,3)
            gn_dot_td, bs, gs, do = signals_pool.shape
            assert do == self.dim_out
            signals_pool = signals_pool.permute(1,0,2,3).reshape(bs,gn_dot_td * gs, do)#64,2040,32
            signals_pool = signals_pool.reshape(bs,self.S_reso if self.S_reso is not None else self.sequence_length,self.Node_num,do)
            signals_pools.append(signals_pool)
        signals_pools = torch.cat(signals_pools, dim = -1)
        assert self.layer_ * self.dim_out == signals_pools.shape[-1]
        out_short = self.Nconv_(signals_pools)
        
        out_gate = torch.sigmoid(self.gate_conv(torch.cat([out_short,out_long],dim=1)))
        
        out = out_short * out_gate + (1 - out_gate) * out_long
        
#         return out,((Kr_masked+Kr_masked_)/2).detach().clone(),((outadj+outadj_)/2).detach().clone()
        return out
