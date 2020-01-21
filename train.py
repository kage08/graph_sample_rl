#GCNQ: Multi Q
import numpy as np
import networkx as nx
import random, io
from rl_alg.dqn import DQNTrainer
from expts.gengraph import random_sbm
from expts.change_baseline import Change
from expts.net_env import NetworkEnv
from ge.models.deepwalk import DeepWalk


import matplotlib.pyplot as plt
from PIL import Image

import torch

from torch.utils.tensorboard import SummaryWriter

from rl_alg.replay import Buffer, PriortizedReplay
from rl_alg.utils import OrnsteinUhlenbeckActionNoise

from expts.influence import influence
import pickle, os
from expts import influence as infl
import gc
import logging, argparse

g_paths = [
    'data/rt/copen.pkl',
    #'data/rt/occupy.pkl'
]

syn = False
ratio = 5
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def arg_parse():
    parser = argparse.ArgumentParser(description='Influence Maxima Arguments')
    parser.add_argument('--logfile',dest='logfile', type=str,default='train.log',
                    help='Logging file')
    parser.add_argument('--logdir', dest='logdir', type=str, default=None,
                    help='Tensorboard LogDir')
    parser.add_argument('--log-level',dest='loglevel', type=int,default=2, choices=[1,2],
                    help='Logging level')
    parser.add_argument('--sample-budget',dest='budget', type=int,default=5,
                    help='Number of queries for sampling')
    parser.add_argument('--extra-seeds',dest='extra_seeds', type=int,default=5,
                    help='Initial number of random seeds')
    parser.add_argument('--prop-prob',dest='prop_probab', type=float,default=0.1,
                    help='Propogation Probability for each Node')
    parser.add_argument('--cpu',dest='cpu', type=int,default=0,
                    help='Number of CPUs to use for influence sampling')
    parser.add_argument('--samples',dest='samples', type=int,default=100,
                    help='Number of samples in Influence Maximization')             
    parser.add_argument('--opt',dest='obj', type=float,default=0,
                    help='Threshold for reward')
    parser.add_argument('--infl-budget',dest='ibudget', type=int,default=10,
                    help='Number of queries during influence(greedy steps)')
    parser.add_argument('--render',dest='render', type=int,default=0,
                    help='1 to Render graphs, 0 to not')
    parser.add_argument('--write',dest='write', type=int,default=1,
                    help='1 to write stats to tensorboard, 0 to not')

    parser.add_argument('--change-seeds',dest='changeSeeds', type=int,default=0,
                    help='1 to change seeds after each episode, 0 to not')
    parser.add_argument('--add-noise',dest='add_noise', type=int,default=1,
                    help='1 to add noise to action 0 to not')
    parser.add_argument('--sep-net',dest='sep_net', type=int,default=0,
                    help='Seperate network rep for actor and critic')

    parser.add_argument('--save-freq',dest='save_every', type=int,default=100,
                    help='Model save frequency')

    parser.add_argument('--eps',dest='num_ep', type=int,default=10000,
                    help='Number of Episodes')
    parser.add_argument('--buffer-size',dest='buff_size', type=int,default=4000,
                    help='Replay buffer Size')

    parser.add_argument('--gcn_layers',dest='gcn_layers', type=int,default=2,
                    help='No. of GN Layers before each pooling')
    parser.add_argument('--num_poolig',dest='num_pooling', type=int,default=1,
                    help='No.pooling layers')
    parser.add_argument('--assign_dim',dest='assign_dim', type=int,default=100,
                    help='pooling hidden dims 1')
    parser.add_argument('--assign_hidden_dim',dest='assign_hidden_dim', type=int,default=150,
                    help='pooling hidden dims 2')

    
    parser.add_argument('--actiondim',dest='action_dim', type=int,default=60,
                    help='Action(Node) Dimensions')
    parser.add_argument('--const_features',dest='const_features', type=int,default=1,
                    help='1 to have constant features')
    parser.add_argument('--inputdim',dest='input_dim', type=int,default=20,
                    help='Node features Dimensions')

    parser.add_argument('--step_reward',dest='nop_reward', type=float,default=0,
                    help='Reward for each step')
    parser.add_argument('--bad_reward',dest='bad_reward', type=float,default=0,
                    help='Reward for each step that is closer to active')
    parser.add_argument('--norm_reward',dest='norm_reward', type=int,default=0,
                    help='Normalize reward with opt')
    parser.add_argument('--max_reward',dest='max_reward', type=int,default=None,
                    help='Normalize reward with opt')
    parser.add_argument('--min_reward',dest='min_reward', type=int,default=None,
                    help='Normalize reward with opt')


    parser.add_argument('--lr',dest='lr', type=float,default=1e-4,
                    help='Learning Rate')
    parser.add_argument('--eta',dest='eta', type=float,default=0.1,
                    help='Target network transfer rate')
    parser.add_argument('--gamma',dest='gamma', type=float,default=0.99,
                    help='Discount rate')
    parser.add_argument('--epsilon',dest='epsilon', type=float,default=0.1,
                    help='Epsilon exploration')
    parser.add_argument('--batch_size',dest='batch_size', type=int,default=100,
                    help='Gradient Update Batch Size')

    parser.add_argument('--use_cuda',dest='use_cuda', type=int,default=1,
                    help='1 to use cuda 0 to not')
    parser.add_argument('--walk_len',dest='walk_len', type=int,default=10,
                    help='Walk Length')

    parser.add_argument('--num_walks',dest='num_walks', type=int,default=80,
                    help='Walk Length')
    parser.add_argument('--win',dest='win', type=int,default=5,
                    help='Window size')
    parser.add_argument('--emb_iters',dest='emb_iters', type=int,default=50,
                    help='Walk Length')

    parser.add_argument('--noise_momentum',dest='noise_momentum', type=float,default=0.15,
                    help='Noise Momentum')
    parser.add_argument('--noise_magnitude',dest='noise_magnitude', type=float,default=0.2,
                    help='Noise Magnitude')
                    
    parser.add_argument('--noise_decay',dest='noise_decay_rate', type=float,default=0.999,
                    help='Noise Decay Rate')
    parser.add_argument('--eta_decay',dest='eta_decay', type=float,default=1.,
                    help='eta Decay Rate')
    parser.add_argument('--alpha_decay',dest='alpha_decay', type=float,default=1.,
                    help='alpha Decay Rate')
    parser.add_argument('--eps_decay',dest='eps_decay_rate', type=float,default=0.999,
                    help='Epsilon Decay Rate')

    parser.add_argument('--sample_times',dest='times_mean', type=int,default=10,
                    help='Number of times to sample objective from fluence algorithm')
    parser.add_argument('--sample_times_env',dest='times_mean_env', type=int,default=5,
                    help='Number of times to sample objective from fluence algorithm for env rewards')
    parser.add_argument('--save_model', dest='save_model', type=str, default='sample_',
                    help='Name of Save model')
    parser.add_argument('--neigh',dest='k', type=int,default=1,
                    help='K nearest for ation')
    
    return parser.parse_args()

random.seed(10)
args = arg_parse()
rg = np.random.RandomState(10)
rg1 = np.random.RandomState(10)
#n = 100
logfile = args.logfile

logging.basicConfig(level=args.loglevel*10, filename=logfile, filemode='w', datefmt='%d-%b-%y %H:%M:%S',
                    format='%(levelname)s - %(asctime)s - %(message)s ')

budget = args.budget
extra_seeds = args.extra_seeds

infl.PROP_PROBAB = args.prop_probab
infl.BUDGET = args.ibudget

from multiprocessing import cpu_count
infl.PROCESSORS = cpu_count() if args.cpu <= 0 else args.cpu

infl.SAMPLES = args.samples
print('Samples icm:', infl.SAMPLES)
render = args.render
write = args.write

changeSeeds = args.changeSeeds

add_noise =args.add_noise

debug = False

save_every = args.save_every

NUM_EP = args.num_ep
BUFF_SIZE = args.buff_size
action_dim = args.action_dim
if args.const_features:
    input_dim = args.input_dim
else:
    input_dim = args.action_dim+2
nop_reward = args.nop_reward

LR = args.lr
eta = args.eta
batch_size = args.batch_size
gcn_layers = args.gcn_layers
num_pooling = args.num_pooling
assign_dim = args.assign_dim
assign_hidden_dim = args.assign_hidden_dim

use_cuda = args.use_cuda

noise_momentum = args.noise_momentum
noise_magnitude = args.noise_magnitude

noise_decay_rate = args.noise_decay_rate
eta_decay = args.eta_decay

times_mean = args.times_mean

noise_param = 1


#generate graph
graphs = []
for g_path in g_paths:
    with open(g_path,'rb') as fl:
        graphs.append(pickle.load(fl))
    print(g_path)
    g = graphs[-1]
    print("Nodes:", len(g))
    print("Edges:", len(g.edges))
    logging.info("Nodes: "+str(len(g))+' Edges: '+str(len(g.edges)))

if args.logdir is None:
    writer = SummaryWriter()
else: 
    writer = SummaryWriter(os.path.join('runs', args.logdir))

#Get best baseline
opts = []
for gp,g in zip(g_paths,graphs):
    opt_obj, local_obj, S_opt = influence(g,g)
    print(gp)
    print('OPT Results:',opt_obj, S_opt)
    logging.info('OPT Results:'+str(opt_obj)+' '+ str(S_opt))
    opts.append(opt_obj)

#Initialize seeds
e_seeds_list = []
for g in graphs:
    e_seeds_list.append(list(rg.choice(len(g), extra_seeds)))
#e_seeds = [31, 171]

logging.debug('Extra Seeds:'+ str(e_seeds_list))
ch = []
for gp,g in zip(g_paths,graphs):
    rs = []
    for _ in range(5):
        change = Change(g, budget=budget*2, seeds=[])
        obj1, local_obj1, S1 = change()
        rs.append(obj1)
    ch.append(np.mean(rs))
    print("Change for %s is %f" % (gp,ch[-1]))
logging.info('Change Results:'+str(obj1)+' '+ str(S1))

if args.obj is not None:
    obj = args.obj

envs = []
for g,seeds in zip(graphs, e_seeds_list):
    env = NetworkEnv(fullGraph=g, seeds=seeds, opt_reward=0, nop_r=args.nop_reward,
                times_mean=args.times_mean_env, bad_reward=args.bad_reward, clip_max=args.max_reward, clip_min=args.min_reward ,normalize=args.norm_reward)
    envs.append(env)
replay = PriortizedReplay(BUFF_SIZE, 10, beta=0.6)

logging.info('State Dimensions: '+str(action_dim))
logging.info('Action Dimensions: '+str(action_dim))


acmodel = DQNTrainer(input_dim=input_dim,state_dim=action_dim, action_dim=action_dim, replayBuff=replay, lr=LR, use_cuda=use_cuda, gamma=args.gamma,
                        eta=eta, gcn_num_layers=gcn_layers, num_pooling=num_pooling, assign_dim=assign_dim, assign_hidden_dim=assign_hidden_dim)

noise = OrnsteinUhlenbeckActionNoise(action_dim, theta=noise_momentum, sigma=noise_magnitude)

#! Doesn't Support nested models
#writer.add_graph(acmodel.actor_critic)
rws = []

def make_const_attrs(graph, input_dim):
    n = len(graph)
    mat = np.ones((n,input_dim))
    #mat = np.random.rand(n,input_dim)
    return mat

def make_env_attrs(n=len(g), input_dim=input_dim, env=env):
    mat1 = np.ones((n,int(input_dim/2)))
    mat2 = np.ones((n,int(input_dim/2)))
    mat1[list(env.active),:] = -1
    mat2[list(env.possible_actions),:] = -1
    return np.concatenate((mat1,mat2),1)

def make_env_attrs_1(env, embs,n=len(g), input_dim=input_dim ):
    mat1 = np.zeros((n,int(action_dim+2)))
    for u in env.active:
        mat1[u,:-2] = embs[u]
        mat1[u,-2] = 1
    for u in env.possible_actions:
        mat1[u,:-2] = embs[u]
        mat1[u,-1] = 1
    return mat1

def get_embeds(g):
    d={}
    for n in g.nodes:
        d[n]=str(n)
    g1 = nx.relabel_nodes(g,d)
    graph_model = DeepWalk(g1,num_walks= args.num_walks, walk_length=args.walk_len, workers=args.cpu if args.cpu>0 else cpu_count())
    
    graph_model.train(window_size = args.win, iter=args.emb_iters, embed_size=action_dim)
    embs = {}
    emb1 = graph_model.get_embeddings()
    for n in emb1.keys():
        embs[int(n)] = emb1[n]

    return embs


k = 10

def get_action(s, emb,nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        value, _ = acmodel.get_values2_(s[0], s[1], emb[v])
        if value>q_vals:
            q_vals = value
            node = v
    return node, q_vals

def get_action_curr1(s, emb,nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        value, _ = acmodel.get_values2(s[0], s[1], emb[v])
        if value>q_vals:
            q_vals = value
            node = v
    return node, q_vals

def get_action_curr2(s, emb,nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        _, value = acmodel.get_values2(s[0], s[1], emb[v])
        if value>q_vals:
            q_vals = value
            node = v
    return node, q_vals

node_attrs = make_const_attrs(g,input_dim)

n_iter = 0
try:
    for ep in range(NUM_EP):
        idx = rg1.randint(len(graphs))
        env = envs[idx]
        g = graphs[idx]
        opt = opts[idx]
        change_score = ch[idx]

        print("Choosing %s"%(g_paths[idx]))

        res = []
        if changeSeeds:
            e_seeds = list(rg.choice(len(g), extra_seeds))
        else:
            e_seeds = e_seeds_list[idx]

        env.reset(seeds=e_seeds)
        node_list = list(env.active.union(env.possible_actions))

        t = env.state
        t0 = len(t)
        s_embs = get_embeds(env.sub)
        if args.const_features:
            s = [node_attrs[node_list], env.state]
        else:
            s = [make_env_attrs_1(env=env, embs=s_embs, n=len(g))[node_list], env.state]
        print('Episode:',ep)
        print('Seeds:', e_seeds)
        tot_r = 0
        tot_r1 = 0
        for stps in range(budget):
            
            
            possible_actions = [node_list.index(x) for x in env.possible_actions]
            
            state_embed, _ = acmodel.get_node_embeddings(nodes_attr=s[0], adj=s[1], nodes=possible_actions)
            l = list(env.possible_actions)
            possible_actions_embed = [s_embs[x] for x in l]

            if rg1.rand() > args.epsilon and (replay.size >batch_size or ep == 0):

                if rg1.rand() > 0.5:
                    actual_action, q = get_action_curr1(s,s_embs, l)
                else:
                    actual_action, q = get_action_curr2(s,s_embs, l)
                proto_action = actual_action_embed = s_embs[actual_action]

                
            else:
                actual_action = rg1.choice(list(env.possible_actions), 1)[0]
                proto_action = actual_action_embed = s_embs[actual_action]

            res.append(actual_action)

            _, r, d, _ = env.step(actual_action)
            
            node_list = list(env.active.union(env.possible_actions))
            t = env.state
            s_embs = get_embeds(env.sub)

            if args.const_features:
                s1 = [node_attrs[node_list], env.state]
            else:
                s1 = [make_env_attrs_1(env=env, embs=s_embs, n=len(g))[node_list], env.state]

            logging.debug('State: '+str(state_embed))
            logging.debug('Action:'+str(proto_action))

            # if last time step or explored entire graph
            if stps == budget-1 or len(env.possible_actions)==0:
                env.step(-1)
                r += env.reward
                d = True
            if d:
                s1[1] *= 0
            tot_r+=r

            t = len(env.state)
            #sub = nx.from_numpy_matrix(t)
            #b,_,_ = influence(sub, sub)
            r1=r+ (1/(len(g)))*(t-t0)
            t0 = t
            if d:
                r1 = r1/opt
            
            tot_r1 += r1

            #TODO: TD Compute
            td = acmodel.td_compute(s,actual_action_embed, r1, s1, s_embs[get_action(s1, s_embs, env.possible_actions)[0]])
            replay.add(s,actual_action_embed, r1, s1,s_embs[get_action(s1, s_embs, env.possible_actions)[0]] , actual_action, td=np.abs(td))

            if (ep==0 and stps<2) or replay.size>batch_size:
                acmodel.gradient_update_sarsa(batch_size=batch_size)
                acmodel.gradient_update_sarsa(batch_size=batch_size)
            torch.cuda.empty_cache()

            n_iter += 1
            if write:
                writer.add_scalar('CriticLoss', acmodel.loss_critic.clone().cpu().data.numpy(), n_iter)

            s = s1
            if d:
                break
        
        print('Critic Loss:', acmodel.loss_critic)
        print('Action:', proto_action)
        print('Value:', q)
        print('Env Reward:', r1)
        print('Reward:', tot_r)
        print('Chosen:', res,'\n')
        logging.info('Episode: '+str(ep)+' Reward: '+ str(tot_r))
        logging.debug('Critic Loss: '+ str(acmodel.loss_critic))
        rws.append(tot_r)

        if write:
            writer.add_scalar('Reward', tot_r, ep+1)
            writer.add_scalar('Influence', env.reward_, ep+1)
            writer.add_scalar('Norm Reward', tot_r1, ep+1)

        
        gc.collect()
            
        
        if ep%save_every == 0:
            #acmodel.save_models(args.save_model)
            torch.save(acmodel,'models/'+args.save_model+str(ep)+'.pth')
        
        noise_param *= max(0.001,noise_decay_rate)
        acmodel.eta = max(0.001, acmodel.eta*eta_decay)
        args.epsilon = max(0.01,args.epsilon*args.eps_decay_rate)


    writer.close()

except KeyboardInterrupt:
    writer.close()