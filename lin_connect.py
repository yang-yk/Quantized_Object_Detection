# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:33:31 2019

@author: yyk17
mail: yyk17@mails.tsinghua.edu.cn
linear quantization of weight in dense and conv2d layer

"""

import torch
from common import front
from device import device
from torch.autograd import Variable as Var



def LinQuant(bit_width=8, with_sign=True, lin_back=True):
    r"""
    Generate a Quantization op using Lin method from Imp. CNN using Log Data Rep.
    
    :param bit_width:  Numbers of bits on this quant op.
    :param with_sign: Add a sign bit to quant op.
    :param lin_back: Use linear back propagation or a quantized gradient.
    
    Forward :

    :math:`Quant(x) = Clamp(Round(x/step)*step,0,2^{FSR}) with step = 2^{FSR-bit\_width}`
    
    BackWard (if not lin_back):

    :math:`grad\_input = sign(grad_output)* Clamp(Round(grad\_output/step)*step,0,2^{FSR})`
    """
    class _LinQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Clip(Round(x/step)*step,0,2^(FSR)) with step = 2^{FSR-bitwight}
            if(bit_width==32):
                return input
            step = torch.FloatTensor([2]).pow(bit_width-1)#.to(input.device())
            if with_sign:
                #return torch.sign(input)*torch.clamp(torch.round(torch.abs(input)*step[0])/step[0], -4,1-1/step[0]) 
                return torch.clamp(torch.round(input*step[0])/step[0], -1,1-1/step[0]) 
            return torch.clamp(torch.round(torch.abs(input)*step[0])/step[0], -1,1-1/step[0])  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if bit_width==32:
                return grad_input
            if lin_back:
                return grad_input
            else:
                print('Error in LinQuan Backward')
    return _LinQuant

#x=torch.FloatTensor([-5,-1])
#x=Var(x,requires_grad=True)
#y=LinQuant().apply(x)
#y.backward(torch.ones_like(y))
#print(x.grad)



def nnQuant(dtype="lin", bit_width=8, with_sign=True, lin_back=True):
    """
    Return a Torch Module fronter with Quantization op inside. Suport Lin and Log quantization.

    :param dtype: Use \'lin\' or \'log\' method.
    :param fsr: Max value of the output.
    :param bit_width:  Numbers of bits on this quant op.
    :param with_sign: Add a sign bit to quant op.
    :param lin_back: Use linear back propagation or a quantized gradient.

    """
    if dtype == "lin":
        return front(LinQuant(bit_width=8, with_sign=True, lin_back=True))
    else:
        raise RuntimeError("Only \'log\' and \'lin\' dtype are supported !")
 



#def QuantDense(input, weight, bias=None, bitwight=8):
#    class LinQuantDense(torch.autograd.Function):
#        @staticmethod
#        def forward(ctx, input, weight, bias=None):
#            #step = torch.FloatTensor([2]).pow(bitwight-1)
#            ctx.save_for_backward(input, weight, bias)
#            #weight_q = torch.clamp(torch.round(weight*step[0])/step[0], -1,1-1/step[0]) 
#            weight_q=LinQuant(bit_width=bitwight).apply(weight)
#            print('weight:',weight_q)
#            #weight_q=weight
#            #print(weight_q)
#            output = torch.nn.functional.linear(input, weight_q, bias)
#            #output=input.mm(weight_q)
#            
#            return output
#
#        @staticmethod
#        def backward(ctx, grad_output):
#            input, weight, bias = ctx.saved_tensors
#            #weight_b = torch.sign(weight)
#            print(grad_output)
#            grad_input = grad_weight = grad_bias = None
#            if ctx.needs_input_grad[0]:
#                #grad_input = grad_output.mm(weight_b)
#                grad_input = grad_output.mm(weight)
#                print('grad_input:',grad_input)
#            if ctx.needs_input_grad[1]:
#                grad_weight = grad_output.t().mm(input)
#                #print(grad_output.t())
#            if bias is not None and ctx.needs_input_grad[2]:
#                grad_bias = grad_output.sum(0).squeeze(0)
#            return grad_input, grad_weight, grad_bias
#
#    return LinQuantDense.apply(input, weight, bias)
#
#
#def QuantConv2d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1, bitwight=8):
#    class _QuantConv2d(torch.autograd.Function):
#        @staticmethod
#        def forward(ctx, input, weight, bias=None):
#            #step = torch.FloatTensor([2]).pow(fsr-bitwight).to(device)
#            ctx.save_for_backward(input, weight, bias)
#            weight_q = LinQuant(bit_width=bitwight).apply(weight)
#            output = torch.nn.functional.conv2d(input, weight_q, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#            return output
#
#        @staticmethod
#        def backward(ctx, grad_output):
#            input, weight, bias = ctx.saved_tensors
#            
#            grad_input = grad_weight = grad_bias = None
#
#            if ctx.needs_input_grad[0]:
#                grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
#            if ctx.needs_input_grad[1]:
#                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
#
#            if bias is not None and ctx.needs_input_grad[2]:
#                grad_bias = grad_output.sum(0).squeeze(0).sum(1).squeeze(1).sum(-1).squeeze(-1)
#
#            if bias is not None:
#                return grad_input, grad_weight, grad_bias
#            else:
#                grad_bias=None
#                return grad_input, grad_weight, grad_bias
#
#    return _QuantConv2d.apply(input, weight, bias)





