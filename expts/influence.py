import networkx as nx
import numpy as np
from icm import sample_live_icm, indicator, make_multilinear_objective_samples
from utils import greedy
from multiprocessing import Process, Manager

PROP_PROBAB = 0.1
BUDGET = 10
PROCESSORS = 8
SAMPLES = 100

def multi_to_set(f,g):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    def f_set(S):
        return f(indicator(S, len(g)))
    return f_set

def influence(graph, full_graph, samples=SAMPLES):
    for u,v in graph.edges():
        graph[u][v]['p']=PROP_PROBAB
    
    def genoptfunction(graph, samples=1000):
        live_graphs = sample_live_icm(graph, samples)
        f_multi = make_multilinear_objective_samples(live_graphs, list(graph.nodes()), list(graph.nodes()), np.ones(len(graph)))
        f_set = multi_to_set(f_multi, graph)
        return f_set
    
    f_set = genoptfunction(graph, samples)
    S, obj = greedy(list(range(len(graph))), BUDGET, f_set)

    f_set1 = genoptfunction(full_graph, samples)
    opt_obj = f_set1(S)

    return opt_obj, obj, S



def parallel_influence(graph, full_graph, times, samples=SAMPLES, influence=influence):

    def influence_wrapper(l,g,fg,s,influence=influence):
        ans = influence(g,fg,s)
        l.append(ans[0])
    
    l = Manager().list()
    processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]
    i=0
    while i<len(processes):
        j = i+PROCESSORS if i+PROCESSORS < len(processes) else len(processes)-1
        ps = processes[i:j]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        i+= PROCESSORS
    l = list(l)
    return np.mean(l)

        
if __name__ == "__main__":
    g = nx.erdos_renyi_graph(100,0.5)
    for u,v in g.edges():
        g[u][v]['p'] = 0.1
    import time
    start = time.time()
    print(parallel_influence(g,g,10, 1000))
    end1 = time.time()
    print('Parallel took', end1-start, 'seconds')
    start = time.time()
    ls = [influence(g,g,100)[0] for _ in range(10)]
    print(np.mean(ls))
    end1 = time.time()
    print('Seq took', end1-start, 'seconds')
