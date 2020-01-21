import networkx as nx
import numpy as np

def random_sbm(n=1000, c=10, p_w=0.1, p_b=0.01, std=30,seed=None):
    '''
    Draws an SBM with given parameters

    n: Number of nodes(Approx.)
    c: Number of communities.
    std: standard deviation of normal distribution of community size.
    p_w: within community edge probability.
    p_b: between communities edge probability.
    '''
    rg = np.random.RandomState(seed)
    com_size = rg.normal(loc=(n/c), scale=std, size=c)
    #com_size = (com_size/np.sum(com_size))*n
    com_size = com_size.astype(int)

    com = [nx.gnp_random_graph(n=x,p=p_w, seed=rg.randint(1000)) for x in com_size]
    
    for i in range(c):
        nx.set_node_attributes(com[i],i,'n')

    G = nx.disjoint_union_all(com)

    for u in G.nodes():
        for v in G.nodes():
            if G.nodes[u]['n'] != G.nodes[v]['n'] and rg.rand()<p_b:
                G.add_edge(u,v)
    
    return G
        
            