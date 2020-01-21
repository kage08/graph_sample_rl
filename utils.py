
def visualize_set(g, S, all_nodes):
    '''
    Draws a visualization of g, with the nodes in S much bigger/colored and the
    nodes in all_nodes drawn in a medium size. Helpful to visualize seed sets/
    conduct sanity checks.
    '''
    import networkx as nx
    node_color = []
    node_size = []
#    g = nx.subgraph(g, all_nodes)
    for v in g.nodes():
        if v in S:
            node_color.append('b')
            node_size.append(300)
        elif v in all_nodes:
            node_color.append('y')
            node_size.append(100)
        else:
            node_color.append('k')
            node_size.append(20)
#    node_size = [300 if v in S else 20 for v in g.nodes()]
    nx.draw(g, node_color = node_color, node_size=node_size)

def visualize_communities(g, part, S = None):
    '''
    Partitions the graph into communities using the "community" package, and 
    draw the communities as distinct groups. Optimally, draw the set of nodes S
    larger.
    '''
    import networkx as nx
    import community
    import numpy as np
    import random
#    part = community.best_partition(g)
#    part = [part[x] for x in g.nodes()]
#    com_names = np.unique(part)
#    communities = []
#    for i,c in enumerate(com_names):
#        communities.append([])
#        communities[i].extend([x for x in g.nodes() if part[x] == c])
#    node_color = part
#    pos = nx.layout.spring_layout(g, k=0.1)
    pos = {}
    centers = [[0, 0], [0, 1.25], [1.25, 0], [1.25, 1.25]]
    for v in g.nodes():
        pos[v] = [centers[part[v]][0] + random.random() - 0.5, centers[part[v]][1] + random.random() - 0.5]
    if not S == None:
        node_colors = []
        node_sizes = []
        for v in g.nodes():
            if v in S:
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append('blue')
                node_sizes.append(50)
            
    nx.draw(g, node_color=node_colors, node_size=node_sizes, pos=pos)
    
#def load_g(netname):
#    import networkx as nx
#    import numpy as np
#    if 'india' in netname:
#        num = netname.split('-')[1]
#        G = np.loadtxt('relations/' + num + '-All2.csv', delimiter=',')
#        g = nx.from_numpy_matrix(G)
#    else:
#        g = nx.read_edgelist(netname + '.txt', nodetype=int)
#    return g

def greedy_icm(g, budget, rr_sets = None, start_S = None):
    '''
    Greedy algorithm specifically for ICM. Currently missing dependency to do
    fast ICM evaluation; will add that later.
    '''
    from rr_icm import make_rr_sets_cython, eval_node_rr
    import heapq
    num_nodes = len(g)
    allowed_nodes = range(num_nodes)
    if rr_sets == None:
        rr_sets = make_rr_sets_cython(g, 500, range(num_nodes))
    if start_S == None:
        S = set()
    else:
        S = start_S
    upper_bounds = [(-eval_node_rr(u, S, num_nodes, rr_sets), u) for u in allowed_nodes]    
    heapq.heapify(upper_bounds)
    starting_objective = 0
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = eval_node_rr(u, S, num_nodes, rr_sets)
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective

def greedy(items, budget, f, init = []):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set(init))
    S  = set(init)
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound up to small epsilon
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective


# TODO: there is a bug here, where we don't require w to be >= 0
def projection_simplex_sort(v, z=1):
    import numpy as np

    if np.sum(v) < z:
        return v

    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def exhaustive_search(items, budget, f):
    '''
    Generic exhaustive search algorithm to select budget number of items to maximize f.
    '''

    import itertools

    S = set()
    val = f(S)
    for subset in itertools.combinations(items, budget):
        this_val = f(subset)
        if this_val > val:
            S = subset
            val = this_val

    return S, val


def greedy_cover(items, c, f):
    '''
    Generic greedy algorithm to find a set with value at least c
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    #greedy selection of K nodes
    while starting_objective < c - 0.0001:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        if len(upper_bounds) == 0:
            if new_total >= c:
                S = S.add(u)
                return S
            else:
                return -1
            
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S

def saturate(items, budget, fs, epsilon):
    '''
    SATURATE algorithm of Krause et al 2008 for robust submodular optimization.
    '''
    from math import ceil
    cmax = fs[0](set(items))
    cmin = 0
    c = (cmax + cmin)/2
    S_best = None
    while cmax - cmin > epsilon:
        f_truncate = lambda S: (1./len(fs))*sum(min((f(S), c)) for f in fs)
        S = greedy_cover(items, c, f_truncate)
        if S == -1 or len(S) > budget:
            cmax = c
            c = (c + cmin)/2
            print('failed', cmax, cmin, c)
        else:
            cmin = c
            c = (c + cmax)/2
            S_best = S
            print('succeed', cmax, cmin, c)

    return S_best

def f_connected_components(S, cc, numscenario=1):
    '''
    S: a set of nodes
    
    cc: a list of the connected components from a graph, each one a set of nodes
    
    numscenario: the number of live edge graphs to divide by
    
    Returns the average number of nodes which lie in the same connected component 
    as a seed node
    '''
    return 1./numscenario * sum([len(c) if not c.isdisjoint(S) else 0 for c in cc])
                
def make_objective_samples(live_edge_graphs, g, weights=None):
    '''
    live_edge_lists: a list of lists. Each list contains a set of edges which are
    live in that instance
    
    g: the underlying graph
    
    Returns: a function f which takes a single argument, a seed set S. f(S) gives
    the average influence of S over the set of live edge graphs.
    '''
    if weights is None:
        weights = [1./len(live_edge_graphs)] * len(live_edge_graphs)

    import networkx as nx
    from functools import partial

    cc_list = [list(nx.connected_components(h)) for h in live_edge_graphs]
    def influence_each_live_edge_graph(S):
        # cc_influences = [f_connected_components(S, cc = cc, numscenario = 1) for cc in cc_list]
        # return [w * cc_i for cc_i, w in zip(cc_influences, weights)]
        return [w * f_connected_components(S, cc = cc, numscenario = 1) \
                for cc, w in zip(cc_list, weights)]

    return influence_each_live_edge_graph

    # ccs = []
    # cc_weights = []
    # for h, w in zip(live_edge_graphs, weights):
    #     cc = list(nx.connected_components(h))
    #     ccs.extend(cc)
    #     cc_weights.extend([w] * len(cc))
    # return partial(f_connected_components, cc = ccs, cc_weights = cc_weights)

def sample_live_icm(g, num_graphs):
    '''
    Returns num_graphs live edge graphs sampled from the ICM on g. Assumes that
    each edge has a propagation probability accessible via g[u][v]['p'].
    '''
    import random
    import networkx as nx
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u,v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u,v)
        live_edge_graphs.append(h)
    return live_edge_graphs

def project_uniform_matroid_boundary(x, k, c=1):
    '''
    Exact projection algorithm of Karimi et al. This is the projection implementation
    that should be used now.
    
    Projects x onto the set {y: 0 <= y <= 1/c, ||y||_1 = k}
    '''
    import numpy as np
    k *= c
    n = len(x)
    x = x.copy()
    alpha_upper = x/c
    alpha_lower = (x*c - 1)/c**2
    S = []
    S.extend(alpha_lower)
    S.extend(alpha_upper)
    S.sort()
    S = np.unique(S)
    h = n
    alpha = min(S) - 1
    m = 0
    for i in range(len(S)):
        hprime = h + (S[i] - alpha)*m
        if hprime < k and k <= h:
            alphastar = (S[i] - alpha)*(h - k)/(h - hprime) + alpha
            result = np.zeros((n))
            for j in range(n):
                if alpha_lower[j] > alphastar:
                    result[j] = 1./c
                elif alpha_upper[j] >= alphastar:
                    result[j] = x[j] - alphastar*c
            return result
        m -= (alpha_lower == S[i]).sum()*(c**2)
        m += (alpha_upper == S[i]).sum()*(c**2)
        h = hprime
        alpha = S[i]
    raise Exception('projection did not terminate')

def project_cvx(x, k):
    '''
    Exact Euclidean projection onto the boundary of the k uniform matroid polytope.
    '''
    from cvxpy import Variable, Minimize, sum_squares, Problem
    import numpy as np
    n = len(x)
    p = Variable(n, 1)
    objective = Minimize(sum_squares(p - x))
    constraints = [sum(p) == k, p >= 0, p <= 1]
    prob = Problem(objective, constraints)
    prob.solve()
    return np.reshape(np.array(p.value), x.shape)

def repeated_stochastic_greedy(items, budget, f, num_repetitions):
    import random
    best_val = -1
    best_S = None
    for i in range(num_repetitions):
        items_rand = []
        for v in items:
            if random.random() < 0.5:
                items_rand.append(v)
        S, val = greedy(items_rand, budget, f)
        if val > best_val:
            best_S = S
            best_val = val
    return best_S, best_val
