import math
import random
import os

import numpy as np

import pdb

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import rl_alg.utils as utls
from diffpool.encoders import GraphConv
from rl_alg.utils import get_torch_device
from collections.abc import Iterable
EPS = 0.003

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None, use_cuda=True, num_aggs=1):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=num_aggs
        self.use_cuda = use_cuda

        if use_cuda:
            self.device = get_torch_device()
        else:
            self.device = torch.device("cpu")

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, use_cuda=self.use_cuda)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, use_cuda=self.use_cuda) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, use_cuda=self.use_cuda)
        return conv_first, conv_block, conv_last


    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(self.device)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        #print(output.size())
        return output


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, num_layers,
            assign_hidden_dim, assign_num_layers=-1, assign_ratio=1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0,
            assign_input_dim=-1, args=None, num_aggs=1, use_cuda=True):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args, num_aggs=num_aggs, use_cuda=use_cuda)
        add_self = not concat
        self.num_pooling = num_pooling
        self.assign_ent = True


        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            

            # next pooling layer
            assign_input_dim = embedding_dim + hidden_dim * (num_layers - 1)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)


        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes=None,compute_loss=True, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        from copy import deepcopy
        # mask
        self.link_loss = torch.zeros(1).to(self.device)
        self.entropy_loss = torch.zeros(1).to(self.device)
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
                
        self.node_embeddings = embedding_tensor.clone()
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = F.softmax(self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask), -1)
            
            # [batch_size x num_nodes x next_lvl_num_nodes]
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask
            if compute_loss:
                self.link_loss += self.loss(adj)
                self.entropy_loss -= (1/adj.size()[-2]) * torch.sum(self.assign_tensor * torch.log(self.assign_tensor))

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)


        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        return output
    
    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        max_num_nodes = adj.size()[1]
        pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
        tmp = pred_adj0
        pred_adj = pred_adj0
        for adj_pow in range(adj_hop-1):
            tmp = tmp @ pred_adj0
            pred_adj = pred_adj + tmp
        pred_adj = torch.min(pred_adj, torch.ones(1).to(self.device))
        #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
        if batch_num_nodes is None:
            num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        else:
            num_entries = np.sum(batch_num_nodes * batch_num_nodes)
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
            self.link_loss[1-adj_mask.byte()] = 0.0

        self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #print('linkloss: ', self.link_loss)
        return self.link_loss
    
class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim=1.):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = utls.fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = utls.fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,100)
		self.fc3.weight.data = utls.fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(100,64)
		self.fc4.weight.data = utls.fanin_init(self.fc4.weight.data.size())

		self.fc5 = nn.Linear(64,action_dim)
		self.fc5.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		action = F.tanh(self.fc5(x))

		action = action * self.action_lim

		return action


class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim+action_dim,400)
		self.fc1.weight.data = utls.fanin_init(self.fc1.weight.data.size())
		
		
		self.fc2 = nn.Linear(400,300)
		self.fc2.weight.data = utls.fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(300,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)
		#self.fc3.weight.data = utls.fanin_init(self.fc3.weight.data.size())

		self.fca1 = nn.Linear(state_dim+action_dim,400)
		self.fca1.weight.data = utls.fanin_init(self.fca1.weight.data.size())
		
		
		self.fca2 = nn.Linear(400,300)
		self.fca2.weight.data = utls.fanin_init(self.fca2.weight.data.size())

		self.fca3 = nn.Linear(300,1)
		self.fca3.weight.data.uniform_(-EPS,EPS)
		#self.fca3.weight.data = utls.fanin_init(self.fca3.weight.data.size())

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		xu = torch.cat((state,action),dim=1)
		x = F.relu(self.fc1(xu))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		x2 = F.relu(self.fca1(xu))
		x2 = F.relu(self.fca2(x2))
		x2 = self.fca3(x2)

		return x, x2
	
	def q1(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		xu = torch.cat((state,action),dim=1)
		x = F.relu(self.fc1(xu))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x
	
	def q2(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		xu = torch.cat((state,action),dim=1)
		x2 = F.relu(self.fca1(xu))
		x2 = F.relu(self.fca2(x2))
		x2 = self.fca3(x2)

		return x2




class GraphTD3(nn.Module):
    def __init__(self, input_dim, node_embedding_dim, graph_embedding_dim, max_num_nodes=10, gcn_num_layers=2, num_pooling=1, assign_dim=40, num_aggs=1, use_cuda=True):
        super(GraphTD3, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.input_dim = input_dim
        if use_cuda:
            self.device = utls.get_torch_device()
        else:
            self.device = torch.device("cpu")
        
        self.graph_embedder = SoftPoolingGcnEncoder(max_num_nodes=max_num_nodes, input_dim=input_dim, hidden_dim=node_embedding_dim,
                                                    embedding_dim=graph_embedding_dim, num_layers=gcn_num_layers, num_pooling=num_pooling,
                                                    assign_hidden_dim=assign_dim, num_aggs=num_aggs).to(self.device)

        self.actor = Actor(graph_embedding_dim, node_embedding_dim).to(self.device)
        self.critic = Critic(graph_embedding_dim, node_embedding_dim).to(self.device)
        self.input_dim = input_dim
        self.node_embedding_dim = node_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim

    
    def actor_forward(self,nodes_attr, adj, adjust_dim=True, convert_torch=True):
        state_embed, _ = self.get_embeddings(nodes_attr, adj, adjust_dim=adjust_dim, convert_torch=convert_torch)
        action = self.actor(state_embed)
        return action
    
    def critic_forward(self,nodes_attr, adj, nodes, adjust_dim=True, convert_torch=True, node_labels=True):
        if node_labels:
            state_embed, action_embed = self.get_embeddings(nodes_attr, adj, nodes, adjust_dim=adjust_dim, convert_torch=convert_torch)

            #if isinstance(nodes, Iterable):
            #    state_embed = state_embed.expand(len(nodes), state_embed.size()[0])
        else:
            state_embed, _ = self.get_embeddings(nodes_attr, adj, adjust_dim=adjust_dim, convert_torch=convert_torch)
            action_embed = nodes

        q_val1, q_val2 = self.critic(state_embed, action_embed)
        return q_val1, q_val2
    
    def critic_forward1(self,nodes_attr, adj, nodes, adjust_dim=True, convert_torch=True, node_labels=True):
        if node_labels:
            state_embed, action_embed = self.get_embeddings(nodes_attr, adj, nodes, adjust_dim=adjust_dim, convert_torch=convert_torch)

            #if isinstance(nodes, Iterable):
            #    state_embed = state_embed.expand(len(nodes), state_embed.size()[0])
        else:
            state_embed, _ = self.get_embeddings(nodes_attr, adj, adjust_dim=adjust_dim, convert_torch=convert_torch)
            action_embed = nodes

        q_val = self.critic.q1(state_embed, action_embed)
        return q_val
    
    def critic_forward2(self,nodes_attr, adj, nodes, adjust_dim=True, convert_torch=True, node_labels=True):
        if node_labels:
            state_embed, action_embed = self.get_embeddings(nodes_attr, adj, nodes, adjust_dim=adjust_dim, convert_torch=convert_torch)

            #if isinstance(nodes, Iterable):
            #    state_embed = state_embed.expand(len(nodes), state_embed.size()[0])
        else:
            state_embed, _ = self.get_embeddings(nodes_attr, adj, adjust_dim=adjust_dim, convert_torch=convert_torch)
            action_embed = nodes

        q_val = self.critic.q2(state_embed, action_embed)
        return q_val
    
    def get_embeddings(self,nodes_attr, adj, nodes=None, adjust_dim=True, convert_torch=True):
        if convert_torch:
            if adjust_dim:
                nodes_attr = np.float32(nodes_attr).reshape((1,)+nodes_attr.shape)
                adj = np.float32(adj).reshape((1,)+adj.shape)

            nodes_attr = Variable(torch.from_numpy(nodes_attr)).to(self.device)
            adj = Variable(torch.from_numpy(adj)).to(self.device)

        state_embed = self.graph_embedder.forward(nodes_attr, adj)[:,-self.graph_embedding_dim:]
        action_embed = None

        if nodes is not None:
            if adjust_dim:
                action_embed = self.graph_embedder.node_embeddings[0,nodes,:self.node_embedding_dim]
            else:
                action_embeds = [self.graph_embedder.node_embeddings[i:i+1,nodes[i],:self.node_embedding_dim] for i in range(len(nodes))]
                action_embed = torch.cat(action_embeds, dim=0)

        return state_embed, action_embed
    

    
    def forward(self,nodes_attr=None, adj=None, nodes=None, adjust_dim=True, convert_torch=True, node_labels=True):
        if nodes_attr is None:
            nodes_attr = np.random.rand(100,self.input_dim)
            adj = np.random.rand(100,100)
        action = self.actor_forward(nodes_attr, adj, adjust_dim, convert_torch)
        q_vals1, q_vals2 = None, None
        if nodes is not None:
            q_vals1, q_vals2 = self.critic_forward(nodes_attr,adj, nodes, adjust_dim, convert_torch, node_labels)

        return action, q_vals1, q_vals2



class DQNTrainer:

    def __init__(self, input_dim, state_dim, action_dim, replayBuff, lr=1e-3, gamma = 0.99, eta=1e-3,
                gcn_num_layers = 2, num_pooling = 1, assign_hidden_dim=40, assign_dim=40, num_aggs = 1,use_cuda=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = input_dim

        self.lr=lr

        self.it = 0
        self.replay = replayBuff
        if use_cuda:
            self.device = utls.get_torch_device()
        else:
            self.device = torch.device("cpu")


        self.actor_critic = GraphTD3(input_dim=input_dim, node_embedding_dim=action_dim, graph_embedding_dim=state_dim,use_cuda=use_cuda,
                                gcn_num_layers=gcn_num_layers, num_pooling=num_pooling, assign_dim=assign_dim, num_aggs=num_aggs, max_num_nodes=assign_hidden_dim).to(self.device)
        
        self.target_actor_critic = GraphTD3(input_dim=input_dim, node_embedding_dim=action_dim, graph_embedding_dim=state_dim,use_cuda=use_cuda,
                                gcn_num_layers=gcn_num_layers, num_pooling=num_pooling, assign_dim=assign_dim, num_aggs=num_aggs, max_num_nodes=assign_hidden_dim).to(self.device)

        #self.actor_critic_opt = torch.optim.Adam(self.actor_critic.parameters(),self.lr, weight_decay=0.001)
        self.critic_opt = torch.optim.Adam(list(self.actor_critic.critic.parameters())+list(self.actor_critic.graph_embedder.parameters()), self.lr, weight_decay=0.001)
        #self.actor_opt = torch.optim.Adam(self.actor_critic.actor.parameters(), self.lr, weight_decay=0.001)


        self.gamma = gamma
        self.eta = eta

        utls.copy_parameters(self.target_actor_critic, self.actor_critic)
    
    
    def get_values(self,nodes_attr, adj, node):
        s = nodes_attr, adj
        q1,q2 =  self.actor_critic.critic_forward(np.array([s[0]], dtype=np.float32),np.array([s[1]], dtype=np.float32),nodes=[0], adjust_dim=False)
        return q1.cpu().data.numpy().ravel()[0], q2.cpu().data.numpy().ravel()[0]
    
    def get_values2(self, nodes_attr, adj, action):
        s = nodes_attr, adj
        a = Variable(torch.from_numpy(action.reshape((1,)+action.shape)).to(self.device))
        q1,q2 =  self.actor_critic.critic_forward(np.array([s[0]], dtype=np.float32),np.array([s[1]], dtype=np.float32),nodes=a, adjust_dim=False, node_labels=False)
        return q1.cpu().data.numpy().ravel()[0], q2.cpu().data.numpy().ravel()[0]

    def get_values_(self,nodes_attr, adj, node):
        s = nodes_attr, adj
        q1,q2 =  self.target_actor_critic.critic_forward(np.array([s[0]], dtype=np.float32),np.array([s[1]], dtype=np.float32),nodes=[0], adjust_dim=False)
        return q1.cpu().data.numpy().ravel()[0], q2.cpu().data.numpy().ravel()[0]
    
    def get_values2_(self, nodes_attr, adj, action):
        s = nodes_attr, adj
        a = Variable(torch.from_numpy(action.reshape((1,)+action.shape)).to(self.device))
        q1,q2 =  self.target_actor_critic.critic_forward(np.array([s[0]], dtype=np.float32),np.array([s[1]], dtype=np.float32),nodes=a, adjust_dim=False, node_labels=False)
        return q1.cpu().data.numpy().ravel()[0], q2.cpu().data.numpy().ravel()[0]
    
    def td_compute(self,s,a,r,s1, a1):
        q1_1, q1_2 = self.get_values2(s[0],s[1],a)
        q1 = min(q1_1,q1_2)
        q2, _ = self.get_values2_(s1[0], s1[1],a1)
        td_error = q1 - (r+self.gamma*q2)
        return td_error


    
    def get_node_embeddings(self, nodes_attr, adj, nodes):
        state_embed, action_embeds = self.target_actor_critic.get_embeddings(nodes_attr=nodes_attr, adj=adj, nodes=nodes)
        state_embed = state_embed.detach().cpu().data.numpy()
        action_embeds = action_embeds.detach().cpu().data.numpy()
        return state_embed, action_embeds
    

    def gradient_update_sarsa(self, batch_size=100):
        s, a, r, s1, a1, ano = self.replay.sample_(batch_size)
        sa = [np.array(x[0], dtype=np.float32) for x in s]
        sb = [np.array(x[1], dtype=np.float32) for x in s]
        sa = [Variable(torch.from_numpy(x.reshape((1,)+x.shape)).to(self.device)) for x in sa]
        sb = [Variable(torch.from_numpy(x.reshape((1,)+x.shape)).to(self.device)) for x in sb]
        a1 = [np.array(x, dtype=np.float32) for x in a1]
        a1 = [Variable(torch.from_numpy(x.reshape((1,)+x.shape)).to(self.device)) for x in a1]
        a = Variable(torch.from_numpy(np.array(a, dtype=np.float32)).to(self.device))
        r = Variable(torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device))
        s1a = [np.array(x[0], dtype=np.float32) for x in s1]
        s1b = [np.array(x[1], dtype=np.float32) for x in s1]
        s1a = [Variable(torch.from_numpy(x.reshape((1,)+x.shape)).to(self.device)) for x in s1a]
        s1b = [Variable(torch.from_numpy(x.reshape((1,)+x.shape)).to(self.device)) for x in s1b]
        #ac = Variable(torch.from_numpy(np.array(a1, dtype=np.float32))).to(self.device)

        self.q_next1 = torch.squeeze(torch.cat([self.target_actor_critic.critic_forward1(x[0],x[1],x[2],adjust_dim=False, convert_torch=False, node_labels=False).detach() for x in zip(s1a,s1b,a1)],dim=0))
        self.q_next2 = torch.squeeze(torch.cat([self.target_actor_critic.critic_forward2(x[0],x[1],x[2],adjust_dim=False, convert_torch=False, node_labels=False).detach() for x in zip(s1a,s1b,a1)],dim=0))
        self.q_next = torch.min(self.q_next1, self.q_next2).detach()
        self.q_expected = r + self.gamma*self.q_next
        self.r = r
        self.q_predicted1 = torch.squeeze(torch.cat([self.actor_critic.forward(x[0],x[1],x[2].view((1,)+x[2].shape),adjust_dim=False, convert_torch=False, node_labels=False)[1] for x in zip(sa, sb, a)], dim=0))
        self.q_predicted2 = torch.squeeze(torch.cat([self.actor_critic.forward(x[0],x[1],x[2].view((1,)+x[2].shape),adjust_dim=False, convert_torch=False, node_labels=False)[2] for x in zip(sa, sb, a)], dim=0))

        #self.loss_critic = F.smooth_l1_loss(q_predicted, q_expected)
        self.loss_critic = F.mse_loss(self.q_predicted1, self.q_expected) + F.mse_loss(self.q_predicted2, self.q_expected)
        #self.actor_critic_opt.zero_grad()
        self.critic_opt.zero_grad()
        self.loss_critic.backward()
        #self.actor_critic_opt.step()
        self.critic_opt.step()

        utls.copy_parameters(self.target_actor_critic, self.actor_critic, self.eta)

    def save_models(self, filename, path='models'):
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path,filename+'.pth')
        torch.save({
            'model':self.actor_critic.state_dict(),
            'replay': self.replay
        }, path)
    
    def load_models(self, filename, path='models'):
        path = os.path.join(path, filename)
        if not os.path.isfile(path):
            raise Exception('No such path:'+path)
        temp = torch.load(path)
        self.actor_critic.load_state_dict(temp['model'], map_location=self.device)

        utls.copy_parameters(self.target_actor_critic, self.actor_critic)
