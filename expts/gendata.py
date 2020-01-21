import pickle
from expts.gengraph import random_sbm
import os
import numpy as np
import networkx as nx

def gen_sbm(num=100,n=1000, c=10, p_w=0.1, p_b=0.01, std=30, output_file="rand_sbm.pkl"):
    l = [random_sbm(n,c,p_w,p_b,std) for i in range(num)]
    with open(os.path.join('data',output_file), 'wb') as f:
        pickle.dump(l,f)

def csv_to_graph(filepath):
        assert os.path.isfile(filepath), 'Not valid file path:'+str(filepath)
        arr = np.genfromtxt(filepath, delimiter=',')
        assert arr.shape[0]==arr.shape[1]
        gr = nx.from_numpy_matrix(arr)
        return gr

def get_graphs(directory='data/relations'):
        assert os.path.isdir(directory), 'Not valid folder path:'+str(directory)
        csv_files = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.csv')]
        for f in csv_files:
                gr = csv_to_graph(f)
                g = f.replace('.csv','.pkl')
                with open(g,'wb') as fl:
                        pickle.dump(gr,fl)