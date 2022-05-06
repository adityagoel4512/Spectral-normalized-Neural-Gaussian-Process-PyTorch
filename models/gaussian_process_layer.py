import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .resnet import ResNetBackbone
from .netresult import NetResult


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
            self,
            out_features: int,
            backbone: nn.Module = ResNetBackbone(input_features=2, num_hidden_layers=5, num_hidden=128,
                                                 dropout_rate=0.1),
            num_inducing: int = 1024,
            momentum: float = 0.9,
            ridge_penalty: float = 1e-6
    ):
        super().__init__()
        self.out_features = out_features
        self.num_inducing = num_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty

        # Random Fourier features (RFF) layer
        random_fourier_feature_layer = nn.Linear(backbone.num_hidden, num_inducing)
        random_fourier_feature_layer.weight.requires_grad_(False)
        random_fourier_feature_layer.bias.requires_grad_(False)
        nn.init.normal_(random_fourier_feature_layer.weight, mean=0.0, std=1.0)
        nn.init.uniform_(random_fourier_feature_layer.bias, a=0.0, b=2 * math.pi)

        self.rff = nn.Sequential(backbone, random_fourier_feature_layer)

        # RFF approximation reduces the GP to a standard Bayesian linear model,
        # with beta being the parameters we wish to estimate by maximising
        # p(beta | D). To this end p(beta) (the prior) is gaussian so the loss
        # can be written as a standard MAP objective
        self.beta = nn.Linear(num_inducing, out_features, bias=False)
        nn.init.normal_(self.beta.weight, mean=0.0, std=1.0)

        # RFF precision and covariance matrices
        self.register_buffer('is_fit', torch.tensor(False))
        self.is_fit = torch.tensor(False)

        self.register_buffer('dataset_passed', torch.tensor(False))
        self.dataset_passed = torch.tensor(False)

        self.register_buffer('max_variance', torch.ones(1))

        self.covariance = Parameter(
            self.ridge_penalty * torch.eye(num_inducing),
            requires_grad=False,
        )

        self.precision_initial = self.ridge_penalty * torch.eye(
            num_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )


    def forward(self, X, with_variance=False, update_precision=False):
        features = torch.cos(self.rff(X))

        if update_precision:
            self.update_precision_(features)

        logits = self.beta(features)

        if not with_variance:
            return NetResult(mean=logits, variance=None)
        else:
            if not self.is_fit:
                raise ValueError(
                    "`compute_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(features[:, None, :], (features @ self.covariance)[:, :, None], ).reshape(-1)
                if not self.dataset_passed:
                    print(torch.max(variances).shape)
                    self.max_variance = torch.max(variances).unsqueeze(dim=0)
                    self.dataset_passed = torch.tensor(True)
                variances = variances / self.max_variance

            return NetResult(logits, variances)

    def reset_precision(self):
        self.precision = self.precision_initial.detach()

    def update_precision_(self, features):
        # This assumes that all classes share a precision matrix like in
        # https://www.tensorflow.org/tutorials/understanding/sngp

        # The original SNGP paper defines precision and covariance matrices on a
        # per class basis, however this can get expensive to compute with large
        # output spaces
        with torch.no_grad():
            if self.momentum < 0:
                # self.precision = identity => self.precision = identity + features.T @ features
                print(f'features: {features.shape}')
                print(f'precision: {self.precision.shape}')
                self.precision = Parameter(self.precision + features.T @ features)
            else:
                self.precision = Parameter(self.momentum * self.precision +
                                           (1 - self.momentum) * features.T @ features)

    def update_precision(self, X):
        with torch.no_grad():
            features = torch.cos(self.rff(X))
            self.update_precision_(features)

    def update_covariance(self):
        if not self.is_fit:
            # The precision matrix is positive definite and so we can use its cholesky decomposition to more
            # efficiently compute its inverse (when num_inducing is large)
            try:
                L = torch.linalg.cholesky(self.precision)
                self.covariance = Parameter(self.ridge_penalty * L.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)
                print('cholesky')
            except:
                self.covariance = Parameter(self.ridge_penalty * self.precision.cholesky_inverse(), requires_grad=False)
                self.is_fit = torch.tensor(True)
                print('standard inversion')
        else:
            print(f'no inversion, already fit')

    def reset_covariance(self):
        self.is_fit = torch.tensor(False)
        self.covariance.zero_()