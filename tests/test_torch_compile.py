"""

Tests for torch.compile

Note: conda install conda-forge::gxx

TODO: 
    - test conda install conda-forge::cxx-compiler
    - check torchtriton for GPU
    
References:
    - https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    - https://github.com/pytorch/pytorch/issues/122094

"""

# global modules
import json
import unittest

import numpy as np
import torch

# torch compile settings
import torch._inductor.config

# Local modules
from torchsurv.loss.cox import neg_partial_log_likelihood as cox

torch._inductor.config.cpp.cxx = ("g++",)

# set seed for reproducibility
torch.manual_seed(42)

# TODO: wrap this in TestCoxSurvivalLossCompile(unittest.TestCase)
if __name__ == "__main__":

    # random data and parameters
    N = 32
    log_hz = torch.randn(N)
    event = torch.randint(low=0, high=2, size=(N,)).bool()
    time = torch.randint(low=1, high=100, size=(N,))

    # compile cox
    ccox = torch.compile(cox)

    loss_cox = cox(log_hz, event, time)
    loss_ccox = ccox(log_hz, event, time)
