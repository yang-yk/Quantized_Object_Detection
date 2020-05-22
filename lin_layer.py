import torch 
import QLayer
import lin_connect


class LinearQuant(torch.nn.Linear):
    @staticmethod
    def convert(other, dtype="lin", bit_width=8):
        if not isinstance(other, torch.nn.Linear):
            raise TypeError("Expected a torch.nn.Linear ! Receive:  {}".format(other.__class__))
        return LinearQuant(other.in_features, other.out_features, False if other.bias is None else True, dtype=dtype,bit_width=bit_width)

    def __init__(self, in_features, out_features, bias=None, dtype="lin", bit_width=8):
        self.bit_width = bit_width
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.weight_op = lin_connect.nnQuant(dtype=dtype, bit_width=bit_width, with_sign=True, lin_back=True)

    def clamp(self):
        self.weight.data.clamp_(-1, 1)

    def train(self, mode=True):
        if self.training==mode:
            return
        self.training=mode
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.weight_op.forward(self.weight).detach())

    def reset_parameters(self):
        super(LinearQuant, self).reset_parameters()
        #torch.nn.init.uniform_(self.weight, 2**(self.fsr-self.bit_width), 2**(self.fsr))
        #self.weight.data.mul_( (torch.rand_like(self.weight)<0.5).type(self.weight.dtype)*2-1)
        #torch.nn.init.xavier_normal_(self.weight,gain=1.0)
        if not self.bias is None:
            self.bias.data.zero_()

    def forward(self, input):
        #print('weight shape:',self.weight.shape)
        out = torch.nn.functional.linear(input, self.weight_op.forward(self.weight), self.bias)
        return out


class QuantConv2d(torch.nn.Conv2d):

    @staticmethod
    def convert(other, bit_width=8, dtype="lin"):
        if not isinstance(other, torch.nn.Conv2d):
            raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
        return QuantConv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
                         padding=other.padding, dilation=other.dilation, groups=other.groups,
                         bias=False if other.bias is None else True, bit_width=bit_width, dtype=dtype)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=None, bit_width=8, dtype="lin"):

        
        self.bit_width = bit_width
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight_op = lin_connect.nnQuant(dtype=dtype, bit_width=bit_width, with_sign=True, lin_back=True)

    def train(self, mode=True):
        if self.training==mode:
            return
        self.training=mode
        if mode:
            self.weight.data.copy_(self.weight.org.data)
        else: # Eval mod
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.org.data.copy_(self.weight.data)
            self.weight.data.copy_(self.weight_op.forward(self.weight).detach())

    def reset_parameters(self):

        super(QuantConv2d, self).reset_parameters()
        #if self.bit_width==32:
            #super(QuantConv2d, self).reset_parameters()
        #torch.nn.init.uniform_(self.weight, 2**(self.fsr-self.bit_width), 2**(self.fsr))
        #self.weight.data.mul_( (torch.rand_like(self.weight)<0.5).type(self.weight.dtype)*2-1)
        #torch.nn.init.kaiming_uniform_(self.weight,mode='fan_in',nonlinearity='relu')
        if not self.bias is None:
            self.bias.data.zero_()

    def clamp(self):
        self.weight.data.clamp_(-1, 1)

    def forward(self, input):
        #print('weight shape:',self.weight.shape)
        #print(self.weight_op.forward(self.weight)*128)
        #print('bias:',self.bias)
        return torch.nn.functional.conv2d(input, self.weight_op.forward(self.weight), bias=self.bias, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups)




