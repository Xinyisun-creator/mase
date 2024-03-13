from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from .utils import get_stats, quantiser_passthrough

from ..quantizers import (
    residual_sign_quantizer,
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
)

# LUTNet
import numpy as np
from typing import Type
from ..quantizers.LUTNet.BaseTrainer import BaseTrainer, LagrangeTrainer
from ..quantizers.LUTNet.MaskBase import MaskBase, MaskExpanded

# LogicNets
from ..quantizers.LogicNets.utils import (
    generate_permutation_matrix,
    get_int_state_space,
    fetch_mask_indices,
)

# LogicNets
from ..quantizers.LogicNets.utils import (
    generate_permutation_matrix,
    get_int_state_space,
    fetch_mask_indices,
)

import torch
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
import sys

sys.path.append("path to torchvision/references/classification/")
from train import evaluate, train_one_epoch, load_data
from pytorch_quantization import quant_modules
from torch import nn

from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn


class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.bypass = False
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None
        self.pruning_masks = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None
            return F.linear(x, w, bias)

    # TODO: implement these as passes
    # def get_quantized_weight(self) -> Tensor:
    #     return self.w_quantizer(self.weight)

    # def get_quantized_weights_with_inputs(self, x: Tensor) -> Tensor:
    #     x = self.x_quantizer(x)
    #     w = self.w_quantizer(self.weight)
    #     bias = self.b_quantizer(self.bias) if self.bias is not None else None
    #     y = F.linear(x, w, bias)
    #     return {
    #         "x": x,
    #         "w": w,
    #         "bias": bias,
    #         "y": y,
    #     }

    # def get_output_bitwidth(self) -> dict:
    #     raise NotImplementedError()


class LinearInteger_tensorRT(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer_tensorRT, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer_tensorRT, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer_tensorRT, width=b_width, frac_width=b_frac_width
        )

from math import ceil, log2

from numpy import ndarray
from torch import Tensor
import torch

from .utils import my_clamp, my_round


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int = None, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if frac_width is None:
        frac_width = width // 2

    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale


class IntegerQuantize_tensorRT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, frac_width: int, is_signed: bool = True):
        return _integer_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def integer_quantizer_tensorRT(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    return IntegerQuantize.apply(x, width, frac_width, is_signed)


def integer_fraction(
    width: int, frac_choices: list, min_value: float, max_value: float
):
    max_half_range = max(abs(min_value), abs(max_value))
    int_width = int(log2(max(0.5, max_half_range))) + 2
    frac_width = max(0, width - int_width)
    frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    import pdb; pdb.set_trace()
    return frac_width
