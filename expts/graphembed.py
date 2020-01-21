import numpy as np
import scipy as sp
import networkx as nx


def nmf_embed(g, n=2, init_W = None, init_H=None):
    from sklearn.decomposition import NMF
    adj = nx.to_numpy_matrix(g)
    #adj = nx.normalized_laplacian_matrix(g)
    if (init_W is None or init_H is None):
        nmf_model = NMF(n_components=n)
    else:
        nmf_model = NMF(n_components=n, init='custom')

    return nmf_model.fit_transform(adj, W=init_W, H=init_H), nmf_model.components_

def nmf_embed1(g, n=2):
    from sklearn.decomposition import TruncatedSVD
    #adj = nx.to_numpy_matrix(g)
    adj = nx.normalized_laplacian_matrix(g)

    nmf_model = TruncatedSVD(n_components=n)
    nmf_model.fit(adj)

    return nmf_model.transform(adj), nmf_model


def node_embed(g, v, nmf_model, adj=None):
    if adj is None:
        adj = nx.to_numpy_matrix(g)
    assert v in g.nodes(), 'No such node in graph:' + str(v)
    return nmf_model.transform(adj[v])