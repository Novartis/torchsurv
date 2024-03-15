import collections
import copy
import sys
from typing import Callable

import torch
from torch import nn


class Momentum(nn.Module):
    r"""
    Survival framework to momentum update learning to decouple batch size during model training.
    Two networks are concurently trained, an online network and a target network. The online network outputs batches
    are concanetaed and used by the target network, so it virtually increase its batchsize.

    The target network is updated using an exponential moving average:

    .. math::

        ( \theta_k \leftarrow m \theta_k + (1 - m) \theta_q)

    Note:
        This code is inspired from MoCo :cite:p:`he2019` and its ability to decouple batch size from training size.

    References:
        .. bibliography::
            :filter: False

            he2019

    """

    def __init__(
        self,
        backbone: nn.Module,
        loss: Callable,
        batchsize: int = 16,
        n: int = 4,
        m: float = 0.999,
    ):
        """
        Initialise the momentum class. Use must provide their model as backbone.

        Args:
            backbone (nn.Module):
                Torch model to be use as backbone. The model must return either one (Cox) or two ouputs (Weibull)
            loss (Callable): Torchsurv loss function (Cox, Weibull)
            batchsize (int, optional):
                Number of samples per batch. Defaults to 16.
            n (int, optional):
                Number of queued batches to be stored for training. Defaults to 4.
            m (float, optional):
                Exponential moving average rate. Defaults to 0.999.

        Examples:
            >>> from torchsurv.loss import cox, weibull
            >>> _ = torch.manual_seed(42)
            >>> n = 4
            >>> params = torch.randn((n, 16))
            >>> events = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
            >>> times = torch.randint(low=1, high=100, size=(n,))
            >>> backbone = torch.nn.Sequential(torch.nn.Linear(16, 1))  # Cox expect one ouput
            >>> model = Momentum(backbone=backbone, loss=cox.neg_partial_log_likelihood)
            >>> model(params, events, times)
            tensor(0.0978, grad_fn=<DivBackward0>)
            >>> model.online(params)  # online network (q) - w/ gradient
            tensor([[-0.7867],
                    [ 0.3161],
                    [-1.2158],
                    [-0.8195]], grad_fn=<AddmmBackward0>)
            >>> model.target(params)  # target network (k) - w/o gradient
            tensor([[-0.7867],
                    [ 0.3161],
                    [-1.2158],
                    [-0.8195]])

        Note:
            `self.encoder_k` is the recommended to be used for inference. It refers to the target network (momentum).
        """
        super().__init__()
        self.m = m
        self.online = copy.deepcopy(backbone)  # q >> online network
        self.target = copy.deepcopy(backbone)  # k >> target network (momentum)
        self._init_encoder_k()
        self.loss = loss

        # survival data structure
        self.survtuple = collections.namedtuple(
            "survival", field_names=["estimate", "event", "time"]
        )

        # Hazards: current batch & memory
        self.memory_q = collections.deque(maxlen=batchsize)  # q >> online network
        self.memory_k = collections.deque(maxlen=batchsize * n)  # k >> target network

    def forward(
        self, x: torch.Tensor, event: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Forward tensor pass over the model

        Args:
            x (torch.Tensor): Input tensors to the backbone model
            event (torch.Tensor): A boolean tensor indicating whether a patient experienced an event.
            time (torch.Tensor): A positive float tensor representing time to event (or censoring time)

        Returns:
            torch.Tensor: A loss tensor for the current batch.

        Examples:
            >>> from torchsurv.loss import cox, weibull
            >>> _ = torch.manual_seed(42)
            >>> n = 128  # samples
            >>> x = torch.randn((n, 16))
            >>> y = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> t = torch.randint(low=1, high=100, size=(n,))
            >>> backbone = torch.nn.Sequential(torch.nn.Linear(16, 1))  # (log hazards)
            >>> model_cox = Momentum(backbone, loss=cox.neg_partial_log_likelihood)  # Cox loss
            >>> with torch.no_grad(): model_cox.forward(x, y, t)
            tensor(2.1366)
            >>> backbone = torch.nn.Sequential(torch.nn.Linear(16, 2))  # (lambda, rho)
            >>> model_weibull = Momentum(backbone, loss=weibull.neg_log_likelihood)  # Weibull loss
            >>> with torch.no_grad(): torch.round(model_weibull.forward(x, y, t), decimals=2)
            tensor(68.0400)

        """

        estimate_q = self.online(x)
        for e in zip(estimate_q, event, time):
            self.memory_q.append(self.survtuple(*[e for e in e]))
        loss = self._bank_loss()
        with torch.no_grad():
            self._update_momentum_encoder()
            estimate_k = self.target(x)
            for e in zip(estimate_k, event, time):
                self.memory_k.append(self.survtuple(*[e for e in e]))
        return loss

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate data with target network

        Args:
            x (torch.Tensor): Input tensors to the backbone model

        Returns:
            torch.Tensor: Predictions from target (momentum) network without augmentation (.eval()) nor gradient.

        Examples:
            >>> from torchsurv.loss import weibull
            >>> _ = torch.manual_seed(42)
            >>> backbone = torch.nn.Sequential(torch.nn.Linear(8, 2))  # Cox expect one ouput
            >>> model = Momentum(backbone=backbone, loss=weibull.neg_log_likelihood)
            >>> model.eval(torch.randn((3, 8)))
            tensor([[ 0.5342,  0.0062],
                    [ 0.6439,  0.7863],
                    [ 0.9771, -0.8513]])

        """

        with torch.no_grad():
            return self.online.eval()(x)

    def _bank_loss(self) -> torch.Tensor:
        """computer the  negative loss likelyhood from memory bank"""

        # Combine current batch and momentum
        bank = self.memory_k + self.memory_q
        assert all(
            x in bank[0]._fields for x in ["estimate", "event", "time"]
        ), "All fields must be present"
        return self.loss(
            torch.stack([mem.estimate.cpu() for mem in bank]).squeeze(),
            torch.stack([mem.event.cpu() for mem in bank]).squeeze(),
            torch.stack([mem.time.cpu() for mem in bank]).squeeze(),
        )

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Exponantial moving average"""
        for param_b, param_m in zip(self.online.parameters(), self.target.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
