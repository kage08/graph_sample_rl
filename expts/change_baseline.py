from expts.net_env import NetworkEnv
from expts.gengraph import random_sbm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from expts.influence import influence



class Change:
    def __init__(self,fullgraph, budget=10, seeds=[], influence_algo=influence):
        self.fullgraph = fullgraph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.fullgraph.nodes())
        self.budget = budget
        self.seeds = seeds
        self.v1 = []
        for v in seeds:
            self.enlarge_graph(v)
    
    def enlarge_graph(self,u):
        assert u<len(self.graph) and u>=0, "Invalid Vertex"

        for v in self.fullgraph.neighbors(u):
            self.graph.add_edge(u,v)
    
    def sample_graph1(self):
        self.v1 = random.sample(range(len(self.graph)),int(self.budget/2))
        for u in self.v1:
            self.enlarge_graph(u)
        
            
    def sample_graph2(self):
        self.v1.extend(self.seeds)
        self.v2 = []
        for u in self.v1:
            s1 = set(self.fullgraph.neighbors(u))-(set(self.v1))
            if len(s1)>0:
                v = random.sample(s1,1)[0]
                self.enlarge_graph(v)
                self.v2.append(v)

    
    def sample_graph(self):
        self.sample_graph1()
        self.sample_graph2()
        
    def get_ans(self):
        return influence(self.graph, self.fullgraph)
    
    def __call__(self):
        self.sample_graph()
        return self.get_ans()


def test():
    plt.ion()

    gr = random_sbm(n=100)
    rand_seeds = 3
    seeds = np.random.choice(len(gr), rand_seeds)

    budget = 10


    env = NetworkEnv(gr, list(seeds), budget)

    env.reset()
    for i in range(budget):
        a = env.sample_action()
        env.step(a)
        if env.done:
            break

    if not env.done:
        env.step(-1)

    print('Global Reward:', env.reward)
    print('Local Reward:', env.local_reward)


class SeqChange(Change):
    def __init__(self,fullgraph, budget=10, seeds=[], influence_algo=influence):
        super(SeqChange, self).__init__(fullgraph=fullgraph, budget=budget, seeds=seeds, influence_algo=influence_algo)
        self.seeds = seeds
        self.v1 = []
    
    def sample_graph2(self):
        self.v1.extend(self.seeds)
        for _ in range(self.budget//2):
            while True:
                while True:
                    u = random.sample(self.v1,1)[0]
                    if len(list(self.fullgraph.neighbors(u)))>0: break
                v = random.sample(list(self.fullgraph.neighbors(u)),1)[0]
                if v not in self.v1:
                    self.enlarge_graph(v)
                    break
            self.v1.append(v)

