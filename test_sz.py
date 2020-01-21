import networkx as nx
import numpy as np
import community
import pickle
from itertools import product


def open_g(path):
    with open(path,'rb') as fl:
        g = pickle.load(fl)
    return g


def get_comm_stats(g, verbose=True):
    n, m = g.number_of_nodes(), g.number_of_edges()
    if verbose:
        print("Nodes: %d, Edges: %d, Average Degree: %f"%(n,m,2*m/n))
    partition = community.best_partition(g)
    communities = set(partition.values())
    nodes = {c: [n for n in partition.keys() if partition[n]==c] for c in communities}
    szs = [len(nodes[c]) for c in nodes.keys()]
    if verbose:
        print(sorted(szs))
        print("Number of communities: %d, Stdev: %f"%(len(communities),np.std(szs)))
    f1 = lambda x : (x*(x-1))/2
    f2 = lambda x,y : x*y

    max_in_edges = np.sum([f1(len(nodes[n])) for n in nodes.keys()])
    max_btw_edges = 0
    for i in communities:
        j = i+1
        while j<len(communities):
            max_btw_edges+= len(nodes[i]) * len(nodes[j])
            j+=1
    
    in_edges, btw_edges = 0,0
    for u,v in g.edges():
        if partition[u] == partition[v]:
            in_edges+= 1
        else:
            btw_edges+=1
    p_in, p_btw = in_edges/max_in_edges,btw_edges/max_btw_edges
    if verbose:
        print("p_in:%f, p_btw:%f"%(p_in, p_btw))
    return sorted(szs), p_in, p_btw


def gen_sim_graph(g, iters = 1):
    graphs = []
    for _ in range(iters):
        szs, p_in, p_btw = get_comm_stats(g, verbose=False)
        gen_gr = nx.random_partition_graph(szs,p_in,p_btw)
        graphs.append(gen_gr)
    return graphs

def gen_rt_graph(g, iters=1):
    graphs = []
    for _ in range(iters):
        szs, p_in, p_btw = get_comm_stats(g, verbose=False)
        gen_gr = nx.Graph()
        gen_gr.add_nodes_from(range(len(g)))
        nt = 0
        par = {}
        for i,s in enumerate(szs):
            par[i] = list()
            for _ in range(s):
                par[i].append(nt)
                nt +=1
        for i,s in enumerate(szs):
            u = par[i][0]
            if len(par[i])>0:
                for v in par[i][1:]: gen_gr.add_edge(u,v)
        
        for s1,s2 in product(par.keys(),par.keys()):
            if s1>=s2: continue
            for u,v in product(par[s1],par[s2]):
                if np.random.rand()<p_btw:
                    gen_gr.add_edge(u,v)
           
        graphs.append(gen_gr)
    return graphs
