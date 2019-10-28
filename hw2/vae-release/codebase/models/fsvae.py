# Copyright (c) 2018 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        
        qm, qv = self.enc.encode(x, y)
        z = ut.sample_gaussian(qm, qv)
        recon_m = self.dec.decode(z, y) # mu: [batch, dim_x] sigma: [batch, 1/10diag(dim_x)]
        dist = Normal(loc=recon_m, scale=1/10*torch.ones_like(x))
        log_prob = dist.log_prob(x).sum(dim=1)
        
        rec = -torch.mean(log_prob, dim=0)
        kl_z = torch.mean(ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v), dim=0)
        
        nelbo = rec + kl_z
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec.decode(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))
