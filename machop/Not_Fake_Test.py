from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import torch

quant_desc = QuantDescriptor(num_bits=4, fake_quant=False)
quantizer = TensorQuantizer(quant_desc)

torch.manual_seed(12345)
x = torch.rand(10, 9, 8, 7)

quant_x = quantizer(x)

import pdb; pdb.set_trace()