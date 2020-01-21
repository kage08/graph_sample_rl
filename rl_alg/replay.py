import numpy as np
from collections import deque
import random

class Buffer:

    def __init__(self, max_size=1000, seed=None):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        random.seed(seed)
    
    @property
    def size(self):
        return len(self.buffer)
    
    def sample(self, ct):
        ct = min(ct, self.size)
        batch = random.sample(self.buffer, ct)
        s = np.float32([x[0] for x in batch])
        a = np.float32([x[1] for x in batch])
        r = np.float32([x[2] for x in batch])
        s1 = np.float32([x[3] for x in batch])
        a1 = np.float32([x[4] for x in batch])

        return s,a,r,s1, a1
    
    def sample_(self, ct):
        ct = min(ct, self.size)
        batch = random.sample(self.buffer, ct)
        s = [x[0] for x in batch]
        a = [x[1] for x in batch]
        r = [x[2] for x in batch]
        s1 = [x[3] for x in batch]
        a1 = [x[4] for x in batch]
        ano = [x[5] for x in batch]

        return s,a,r,s1, a1, ano
    
    def add(self, s,a,r,s1, a1=None, ano=None):
        arr= [s,a,r,s1, a1, ano]
        self.buffer.append(arr)

class PriortizedReplay(Buffer):
    def __init__(self,max_size=1000, seed=None, beta=1., eps = 0.1):
        super(PriortizedReplay, self).__init__(max_size, seed)
        self.beta = beta
        self.probs = deque(maxlen=self.max_size)
        self.rg = np.random.RandomState(seed)
        self.eps = eps
    
    def add(self,s,a,r,s1, a1=None, ano=None, td=0):
        arr= [s,a,r,s1, a1, ano]
        self.probs.append(td+self.eps)
        self.buffer.append(arr)
    
    def sample(self, ct):
        ct = min(ct, self.size)
        probs = np.array(self.probs)
        probs = probs ** self.beta
        probs = probs/probs.sum()
        idx = [self.rg.choice(self.size, p=probs) for _ in range(ct)]
        s = np.float32([self.buffer[i][0] for i in idx])
        a = np.float32([self.buffer[i][1] for i in idx])
        r = np.float32([self.buffer[i][2] for i in idx])
        s1 = np.float32([self.buffer[i][3] for i in idx])
        a1 = np.float32([self.buffer[i][4] for i in idx])

        return s,a,r,s1,a1
    
    def sample_(self,ct):
        ct = min(ct, self.size)
        probs = np.array(self.probs)
        probs = probs.argsort() +1
        probs = (1/probs)
        probs = probs ** self.beta
        probs = probs/probs.sum()
        idx = [self.rg.choice(self.size, p=probs) for _ in range(ct)]
        s = [self.buffer[i][0] for i in idx]
        a = [self.buffer[i][1] for i in idx]
        r = [self.buffer[i][2] for i in idx]
        s1 =[self.buffer[i][3] for i in idx]
        a1 =[self.buffer[i][4] for i in idx]
        ano =[self.buffer[i][5] for i in idx]

        return s,a,r,s1,a1,ano

