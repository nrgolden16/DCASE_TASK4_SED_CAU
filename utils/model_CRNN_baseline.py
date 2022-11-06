#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res


# class Dynamic_conv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
#                  temperature=31, pool_dim='freq'):
#         super(Dynamic_conv2d, self).__init__()

#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.pool_dim = pool_dim

#         self.n_basis_kernels = n_basis_kernels
#         self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
#                                      temperature, pool_dim)

#         self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
#                                    requires_grad=True)

#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
#         else:
#             self.bias = None

#         for i in range(self.n_basis_kernels):
#             nn.init.kaiming_normal_(self.weight[i])

#     def forward(self, x): #x size : [bs, in_chan, frames, freqs]
#         if self.pool_dim in ['freq', 'chan']:
#             softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)    # size : [bs, n_ker, 1, frames, 1]
#         elif self.pool_dim == 'time':
#             softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)    # size : [bs, n_ker, 1, 1, freqs]
#         elif self.pool_dim == 'both':
#             softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # size : [bs, n_ker, 1, 1, 1]

#         batch_size = x.size(0)

#         aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size) # size : [n_ker * out_chan, in_chan]

#         if self.bias is not None:
#             aggregate_bias = self.bias.view(-1)
#             output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
#         else:
#             output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
#             # output size : [bs, n_ker * out_chan, frames, freqs]

#         output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
#         # output size : [bs, n_ker, out_chan, frames, freqs]

#         if self.pool_dim in ['freq', 'chan']:
#             assert softmax_attention.shape[-2] == output.shape[-2]
#         elif self.pool_dim == 'time':
#             assert softmax_attention.shape[-1] == output.shape[-1]

#         output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

#         return output


# class attention2d(nn.Module):
#     def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
#         super(attention2d, self).__init__()
#         self.pool_dim = pool_dim
#         self.temperature = temperature

#         hidden_planes = int(in_planes / 4)

#         if hidden_planes < 4:
#             hidden_planes = 4

#         if not pool_dim == 'both':
#             self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
#             self.bn = nn.BatchNorm1d(hidden_planes)
#             self.relu = nn.ReLU(inplace=True)
#             self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
#             for m in self.modules():
#                 if isinstance(m, nn.Conv1d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#                 if isinstance(m, nn.BatchNorm1d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#         else:
#             self.fc1 = nn.Linear(in_planes, hidden_planes)
#             self.relu = nn.ReLU(inplace=True)
#             self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

#     def forward(self, x): #x size : [bs, chan, frames, freqs]
#         if self.pool_dim == 'freq':
#             x = torch.mean(x, dim=3)  #x size : [bs, chan, frames]
#         elif self.pool_dim == 'time':
#             x = torch.mean(x, dim=2)  #x size : [bs, chan, freqs]
#         elif self.pool_dim == 'both':
#             # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
#             x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
#         elif self.pool_dim == 'chan':
#             x = torch.mean(x, dim=1)  #x size : [bs, freqs, frames]

#         if not self.pool_dim == 'both':
#             x = self.conv1d1(x)               #x size : [bs, hid_chan, frames]
#             x = self.bn(x)
#             x = self.relu(x)
#             x = self.conv1d2(x)               #x size : [bs, n_ker, frames]
#         else:
#             x = self.fc1(x)               #x size : [bs, hid_chan]
#             x = self.relu(x)
#             x = self.fc2(x)               #x size : [bs, n_ker]

#         return F.softmax(x / self.temperature, 1)


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( avg_pool ) 
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( max_pool )   
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale 

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class ChannelPool(nn.Module):  
#     def forward(self, x):
#         return torch.cat( (torch.max(x,dim=1)[0].unsqueeze(1), torch.mean(x,dim=1).unsqueeze(1)), dim=1 )

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
# #         scale = scale.permute(0,2,1,3)
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         return x_out
    
    
    
    
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
#         self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, (in_tensor.size(2), in_tensor.size(3)), stride=(in_tensor.size(2), in_tensor.size(3)) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att =1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

    
    
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
#         super(ChannelGate, self).__init__()
#         self.gate_c = nn.Sequential()
#         self.gate_c.add_module( 'flatten', Flatten() )
#         gate_channels = [gate_channel]
#         gate_channels += [gate_channel // reduction_ratio] * num_layers
#         gate_channels += [gate_channel]
#         for i in range( len(gate_channels) - 2 ):
#             self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
#             self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
#         self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
#     def forward(self, in_tensor):
#         avg_pool = F.avg_pool2d( in_tensor, (in_tensor.size(2), in_tensor.size(3)), stride=(in_tensor.size(2), in_tensor.size(3)) )
#         max_pool = F.max_pool2d( in_tensor, (in_tensor.size(2), in_tensor.size(3)), stride=(in_tensor.size(2), in_tensor.size(3)) )
        
#         avg_att = self.gate_c( avg_pool )
#         max_att = self.gate_c( max_pool )
        
#         att_sum = avg_att + max_att
        
#         return F.sigmoid(att_sum).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)   #F.sigmoid

# class SpatialGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
#         super(SpatialGate, self).__init__()
#         self.gate_s = nn.Sequential()
#         self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
#         self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
#         self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
#         for i in range( dilation_conv_num ):
#             self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
# 						padding=dilation_val, dilation=dilation_val) )
#             self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
#             self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
#         self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
#     def forward(self, in_tensor):
#         return self.gate_s( in_tensor ).expand_as(in_tensor)
# class BAM(nn.Module):
#     def __init__(self, gate_channel):
#         super(BAM, self).__init__()
#         self.channel_att = ChannelGate(gate_channel)
#         self.spatial_att = SpatialGate(gate_channel)
#     def forward(self,in_tensor):
#         att = F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )   ## 1+
#         return att * in_tensor


class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 atte_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
#             if DY_layers[i] == 1:
#                 cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
#                                                                    n_basis_kernels=n_basis_kernels,
#                                                                    temperature=temperature, pool_dim=pool_dim))
#             else:
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))
                
                
            if atte_layers[i] == 1:
                cnn.add_module("attention{0}".format(i), BAM(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def forward(self, x): #input size : [bs, freqs, frames]
        #cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1) #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
#             print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(bs, frame, ch*freq)   # x.contiguous.view(bs, frame, ch*freq) 
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        #rnn
        x = self.rnn(x) #x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak



