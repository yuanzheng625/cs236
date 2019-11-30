import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedLinear(nn.Linear):
	"""Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""
	def __init__(self, input_size, output_size, mask):
		super().__init__(input_size, output_size)
		self.register_buffer('mask', mask)

	def forward(self, x):
		return F.linear(x, self.mask * self.weight, self.bias)


class PermuteLayer(nn.Module):
	"""Layer to permute the ordering of inputs.

	Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
	"""
	def __init__(self, num_inputs):
		super(PermuteLayer, self).__init__()
		self.perm = np.array(np.arange(0, num_inputs)[::-1])

	def forward(self, inputs):
		return inputs[:, self.perm], torch.zeros(
			inputs.size(0), 1, device=inputs.device)

	def inverse(self, inputs):
		return inputs[:, self.perm], torch.zeros(
			inputs.size(0), 1, device=inputs.device)


class MADE(nn.Module):
	"""Masked Autoencoder for Distribution Estimation.
	https://arxiv.org/abs/1502.03509

	Uses sequential ordering as in the MAF paper.
	Gaussian MADE to work with real-valued inputs"""
	def __init__(self, input_size, hidden_size, n_hidden):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_hidden = n_hidden

		masks = self.create_masks()

		# construct layers: inner, hidden(s), output
		self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
		self.net += [nn.ReLU(inplace=True)]
		# iterate over number of hidden layers
		for i in range(self.n_hidden):  
			self.net += [MaskedLinear(
				self.hidden_size, self.hidden_size, masks[i+1])]
			self.net += [nn.ReLU(inplace=True)]
		# last layer doesn't have nonlinear activation
		self.net += [MaskedLinear(
			self.hidden_size, self.input_size * 2, masks[-1].repeat(2,1))]
		self.net = nn.Sequential(*self.net)

	def create_masks(self):
		"""
		Creates masks for sequential (natural) ordering. 
		"""
		masks = []
		input_degrees = torch.arange(self.input_size)
		degrees = [input_degrees] # corresponds to m(k) in paper

		# iterate through every hidden layer
		for n_h in range(self.n_hidden+1):
			degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
		degrees += [input_degrees % self.input_size - 1]
		self.m = degrees

		# output layer mask
		for (d0, d1) in zip(degrees[:-1], degrees[1:]):
			masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

		return masks

	def forward(self, z):
		"""
		Run the forward mapping (z -> x) for MAF through one MADE block.
		:param z: Input noise of size (batch_size, self.input_size)
		:return: (x, log_det). log_det should be 1-D (batch_dim,)
		"""
		x = torch.zeros_like(z)
	
		# YOUR CODE STARTS HERE
		# for sampling use, so sequential 
		for i in range(self.input_size): #alpha_i is log_sigma
			net_out = self.net(x)
			mu, alpha = net_out[:,:self.input_size], net_out[:,self.input_size:]
			x[:, i] = z[:, i]*torch.exp(alpha[:, i])+mu[:,i]
		log_det = None
		#raise NotImplementedError
		# YOUR CODE ENDS HERE

		return x, log_det

	def inverse(self, x):
		"""
		Run one inverse mapping (x -> z) for MAF through one MADE block.
		:param x: Input data of size (batch_size, self.input_size)
		:return: (z, log_det). log_det should be 1-D (batch_dim,)
		"""
		# YOUR CODE STARTS HERE
		net_out = self.net(x)
		mu, alpha = net_out[:,:self.input_size], net_out[:,self.input_size:]
		z = (x-mu)/torch.exp(alpha)
		log_det = -torch.sum(alpha, dim=1)
		#raise NotImplementedError
		# YOUR CODE ENDS HERE

		return z, log_det


class MAF(nn.Module):
	"""
	Masked Autoregressive Flow, using MADE layers.
	https://arxiv.org/abs/1705.07057
	"""
	def __init__(self, input_size, hidden_size, n_hidden, n_flows):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_hidden = n_hidden
		self.n_flows = n_flows
		self.base_dist = torch.distributions.normal.Normal(0,1)
		
		# need to flip ordering of inputs for every layer
		nf_blocks = []
		for i in range(self.n_flows):
			nf_blocks.append(
				MADE(self.input_size, self.hidden_size, self.n_hidden))
			nf_blocks.append(PermuteLayer(self.input_size))  # permute dims
		self.nf = nn.Sequential(*nf_blocks)

	def log_probs(self, x):
		"""
		Obtain log-likelihood p(x) through one pass of MADE
		:param x: Input data of size (batch_size, self.input_size)
		:return: log_prob. This should be a Python scalar.
		"""
		# YOUR CODE STARTS HERE
		sum_log_det = torch.zeros([x.size(0)], device=x.device)
		for flow in self.nf:
			x, log_det = flow.inverse(x)
			if len(log_det.size())==2:
				log_det = log_det.sum(dim=1)
			sum_log_det+= log_det
		
		sum_log_prob = torch.zeros([x.size(0)], device=x.device)
		for i in range(self.input_size):
			sum_log_prob+= self.base_dist.log_prob(x[:,i])
		
		log_prob = sum_log_det + sum_log_prob
		log_prob = log_prob.mean()
		#raise NotImplementedError
		# YOUR CODE ENDS HERE

		return log_prob

	def loss(self, x):
		"""
		Compute the loss.
		:param x: Input data of size (batch_size, self.input_size)
		:return: loss. This should be a Python scalar.
		"""
		return -self.log_probs(x)

	def sample(self, device, n):
		"""
		Draw <n> number of samples from the model.
		:param device: [cpu,cuda]
		:param n: Number of samples to be drawn.
		:return: x_sample. This should be a numpy array of size (n, self.input_size)
		"""
		with torch.no_grad():
			x_sample = torch.randn(n, self.input_size).to(device)
			for flow in self.nf[::-1]:
				x_sample, log_det = flow.forward(x_sample)
			x_sample = x_sample.view(n, self.input_size)
			x_sample = x_sample.cpu().data.numpy()

		return x_sample