import torch
import numpy as np
from torch.distributions import Normal

def get_torch_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    n = Normal(0,v)
    #return torch.Tensor(size).uniform_(-v, v)
    return n.sample(size)

def copy_parameters(target, source, gamma=1.):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.-gamma) + param.data*gamma)

def save_network(net, filename):
    torch.save(net, filename+'pth')

def save_parameters(net, filename):
    torch.save(net.state_dict(), filename+'pth')

def save_param_opt(net, opt, filename):
    torch.save({'params': net.state_dict(),
                'opt':opt.state_dict()}, filename+'pth')

class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X