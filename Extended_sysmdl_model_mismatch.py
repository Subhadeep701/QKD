import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

##############################
### Pink Noise Generator   ###
##############################
def generate_pink_noise(T, dim=1, per_step_std=1.0):
    """
    Generate pink (1/f) noise with per-step standard deviation per_step_std.
    Ensures that instantaneous variance matches Q for fair UKF comparison.

    Args:
        T: Number of timesteps
        dim: State dimension
        per_step_std: Tensor of shape [dim] or scalar

    Returns:
        pink: Tensor of shape (dim, T)
    """
    uneven = T % 2
    X = torch.randn(dim, T // 2 + 1 + uneven, dtype=torch.cfloat)
    S = torch.arange(1, X.shape[1] + 1, dtype=torch.float32)
    S = 1.0 / torch.sqrt(S)
    X = X * S

    # Mirror spectrum
    if uneven:
        X = torch.cat([X, torch.conj(X[:, :-1].flip(dims=[1]))], dim=1)
    else:
        X = torch.cat([X, torch.conj(X[:, 1:-1].flip(dims=[1]))], dim=1)

    pink = torch.fft.ifft(X).real
    pink = pink - pink.mean(dim=1, keepdim=True)
    pink = pink / pink.std(dim=1, keepdim=True)

    # Scale per state dimension
    if torch.is_tensor(per_step_std) and per_step_std.numel() > 1:
        pink = pink * per_step_std.view(-1, 1)
    else:
        pink = pink * per_step_std

    return pink

###############################################
### Class: System Model for Non-linear Cases ###
###############################################
class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None, noise_type='white'):
        self.f = f
        self.m = m
        self.Q = Q
        self.h = h
        self.n = n
        self.R = R
        self.T = T
        self.T_test = T_test

        self.prior_Q = prior_Q if prior_Q is not None else torch.eye(self.m)
        self.prior_Sigma = prior_Sigma if prior_Sigma is not None else torch.zeros((self.m, self.m))
        self.prior_S = prior_S if prior_S is not None else torch.eye(self.n)

        assert noise_type in ['white', 'pink'], "noise_type must be 'white' or 'pink'"
        self.noise_type = noise_type

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q
        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        self.x = torch.zeros(size=[self.m, T])
        self.y = torch.zeros(size=[self.n, T])
        self.x_prev = self.m1x_0
        xt = self.x_prev

        if self.noise_type == 'pink':
            process_pink = generate_pink_noise(T=T, dim=self.m, per_step_std=torch.sqrt(torch.diag(Q_gen)))
            obs_pink = generate_pink_noise(T=T, dim=self.n, per_step_std=torch.sqrt(torch.diag(R_gen)))

        for t in range(T):
            xt = self.f(self.x_prev)
            if self.noise_type == 'pink':
                xt += process_pink[:, t].view(self.m, 1)

            yt = self.h(xt)
            if self.noise_type == 'pink':
                yt += obs_pink[:, t].view(self.n, 1)

            if self.noise_type == 'white':
                if self.m > 1:
                    xt += MultivariateNormal(torch.zeros(self.m), covariance_matrix=Q_gen).rsample().view(self.m, 1)
                else:
                    xt += torch.normal(mean=0.0, std=torch.sqrt(Q_gen)).view(1, 1)

                if self.n > 1:
                    yt += MultivariateNormal(torch.zeros(self.n), covariance_matrix=R_gen).rsample().view(self.n, 1)
                else:
                    yt += torch.normal(mean=0.0, std=torch.sqrt(R_gen)).view(1, 1)

            self.x[:, t] = xt.squeeze()
            self.y[:, t] = yt.squeeze()
            self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        # Initialize batch
        if randomInit:
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                for i in range(size):
                    self.m1x_0_rand[i, :, 0] = torch.rand(self.m) * args.variance
            elif args.distribution == 'normal':
                for i in range(size):
                    self.m1x_0_rand[i, :, 0] = MultivariateNormal(self.m1x_0.view(-1), self.m2x_0).rsample()
            else:
                raise ValueError('Unsupported distribution')
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)
        else:
            self.Init_batched_sequence(self.m1x_0.expand(size, -1, -1), self.m2x_0)

        self.Input = torch.empty(size, self.n, T)
        self.Target = torch.empty(size, self.m, T)
        self.x_prev = self.m1x_0_batch

        if self.noise_type == 'pink':
            process_pink_batch = torch.stack([
                generate_pink_noise(T=T, dim=self.m, per_step_std=torch.sqrt(torch.diag(self.Q))) for _ in range(size)
            ])
            obs_pink_batch = torch.stack([
                generate_pink_noise(T=T, dim=self.n, per_step_std=torch.sqrt(torch.diag(self.R))) for _ in range(size)
            ])

        for t in range(T):
            xt = self.f(self.x_prev)
            if self.noise_type == 'pink':
                xt += process_pink_batch[:, :, t].view(size, self.m, 1)

            yt = self.h(xt)
            if self.noise_type == 'pink':
                yt += obs_pink_batch[:, :, t].view(size, self.n, 1)

            if self.noise_type == 'white':
                if self.m > 1:
                    eq = MultivariateNormal(torch.zeros(self.m), self.Q).rsample((size,)).view(size, self.m, 1)
                    xt += eq
                else:
                    xt += torch.normal(mean=0.0, std=torch.sqrt(self.Q), size=(size, 1, 1))

                if self.n > 1:
                    er = MultivariateNormal(torch.zeros(self.n), self.R).rsample((size,)).view(size, self.n, 1)
                    yt += er
                else:
                    yt += torch.normal(mean=0.0, std=torch.sqrt(self.R), size=(size, 1, 1))

            self.Target[:, :, t] = xt.squeeze(2)
            self.Input[:, :, t] = yt.squeeze(2)
            self.x_prev = xt
