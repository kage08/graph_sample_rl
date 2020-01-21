import numpy as np
import networkx as nx
from copy import copy
from expts.influence import influence, parallel_influence
import matplotlib.pyplot as plt
import random

class NetworkEnv(object):
    '''
    Environment for network discovery;
    Added negative penalty

    fullGraph: actual graph
    seeds: list of initial seed nodes
    max_T: max length of episode
    influence_algo: function that inputs the discovered graph and does influence maximization to give reward
    '''

    def __init__(self,fullGraph, seeds=[], max_T=100, influence_algo=influence, opt_reward = None, nop_r=0, times_mean=1, bad_reward=0, normalize = False, clip_min = None, clip_max = None):
        self.fullGraph = fullGraph
        self.seeds = seeds
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.fullGraph.nodes())
        self.done = False
        self.T = 0
        self.max_T = max_T
        self.reward = 0
        self.influence_algo = influence_algo
        if opt_reward is None:
            self.opt_reward, _, self.opt_ans = self.influence_algo(self.fullGraph, full_graph=self.fullGraph)
        else:
            self.opt_reward = opt_reward
        self.bad_reward = bad_reward
        self.times_mean  = times_mean

        self.normalize = normalize

        self.active = set(seeds)
        self.possible_actions = set()

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.no_reward = nop_r

        for v in self.active:
            self.enlarge_graph(v)

        self.ans = None
    
    def reset(self, seeds=None):
        if not seeds is None:
            self.seeds = seeds
        
        self.active = set(self.seeds)
        self.possible_actions = set()
        self.reward = 0
        self.T = 0
        self.done = False

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.fullGraph.nodes())


        for v in self.active:
            self.enlarge_graph(v)

   
    def enlarge_graph(self, u):
        '''
        Explore from given node to find neighbors
        '''
        assert u in self.graph.nodes(), "No such node: "+str(u)

        self.active.add(u)

        for v in self.fullGraph.neighbors(u):
            self.graph.add_edge(u,v)
            if not v in self.active:
                self.possible_actions.add(v)
        
        if u in self.possible_actions:
            self.possible_actions.remove(u)
        


    def step(self, action):
        '''
        Action to discover a node.
        If action is -1, we stop discovery and return reward as influence maximization output else reward is 0
        '''
        if action == -1:
            self.done = True
            self.reward_ = parallel_influence(self.graph, full_graph=self.fullGraph,
                                influence=self.influence_algo, times=self.times_mean)
            if self.normalize:
                self.reward = self.reward_/self.opt_reward
            else:
                self.reward = self.reward_ - self.opt_reward
            if self.clip_max is not None or self.clip_min is not None:
                self.reward = np.clip(self.reward, self.clip_min, self.clip_max)
            return self.graph, self.reward, self.done, None
        
        if action in self.possible_actions:
            self.enlarge_graph(action)
            self.reward = self.no_reward
            self.T += 1
            if self.T > self.max_T:
                self.done = True
            return self.graph, self.reward, self.done, None
        if action in self.active:
            self.reward = self.bad_reward
            self.T += 1
            if self.T > self.max_T:
                self.done = True
            self.reward_ = parallel_influence(self.graph, full_graph=self.fullGraph,
                                influence=self.influence_algo, times=self.times_mean)
            if self.normalize:
                self.reward = self.reward_/self.opt_reward
            else:
                self.reward = self.reward_ - self.opt_reward
            if self.clip_max is not None or self.clip_min is not None:
                self.reward = np.clip(self.reward, self.clip_min, self.clip_max)
            return self.graph, self.reward, self.done, None

        else:
            raise Exception("Wrong Action:"+str(action)+"\nPossible Actions:"+str(self.possible_actions))

    @property
    def state(self):
        self.sub = nx.subgraph(self.graph, self.active.union(self.possible_actions))
        return nx.to_numpy_matrix(self.sub)
    
    def draw_curr(self, pos=None):
        '''
        Draw discovered graph
        '''
        plt.figure(1,figsize=(30,30))
        plt.clf()
        self.sub = nx.subgraph(self.graph, self.active.union(self.possible_actions))
        if pos is None:
            nx.draw_random(self.sub,with_labels = True)
        else:
            nx.draw(self.sub,with_labels = True, pos=pos)
    
    def sample_action(self):
        if len(self.possible_actions)>0:
            return random.sample(self.possible_actions,1)[0]
        else:
            return -1
