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
#         return torch.cat( (torch.max(x,dim=2)[0].unsqueeze(1), torch.mean(x,dim=2).unsqueeze(1)), dim=1 )

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
#         scale = scale.permute(0,2,1,3)
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
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

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



# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
#         super(ChannelGate, self).__init__()
# #         self.gate_activation = gate_activation
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
#         return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

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
#         att = F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
#         return att * in_tensor

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class TimeGate(nn.Module):
    def __init__(self, time_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(TimeGate, self).__init__()
        self.time_channels = time_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(time_channels, int(time_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(time_channels // reduction_ratio), time_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        x_p = x.permute(0,2,1,3)   # ( bs, time, chan, freq)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x_p, (x_p.size(2), x_p.size(3)), stride=(x_p.size(2), x_p.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x_p, (x_p.size(2), x_p.size(3)), stride=(x_p.size(2), x_p.size(3)))
                channel_att_raw = self.mlp( max_pool )
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x_p, 2, (x_p.size(2), x_p.size(3)), stride=(x_p.size(2), x_p.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(1).unsqueeze(3).expand_as(x)
        return x * scale
    
    
    
# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, time_channels, reduction_ratio=16, pool_types=['avg', 'max']):  #no_spatial=False
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.TimeGate = TimeGate(time_channels)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.TimeGate(x_out)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
        return x_out 


class ResidualConvBlock(nn.Module):
    def __init__(self,nIn,nOut,timedim):
        super(ResidualConvBlock, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.BatchNorm2d(nIn, eps=0.001, momentum=0.99),
            ContextGating(nIn),
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
#             ContextGating(nOut),
        )
        self.convblock2 = nn.Sequential(
            nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
            ContextGating(nOut),
            nn.Conv2d(nOut, nOut, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
#             ContextGating(nOut),
        )
        self.cbam= CBAM(nOut,timedim)
        if nIn != nOut:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(nIn, nOut, kernel_size=1,stride=1, padding=0),
            )
        else:
            self.residual_conv = nn.Identity()
    def forward(self,x):
        residual = x
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.cbam(x)

        return x + self.residual_conv(residual)
        




class CNN(nn.Module):
    def __init__(
        self,n_ch=1
    ):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        self.cg1 = ContextGating(32)
        self.dropout1 = nn.Dropout(0.5)
        self.avgpool1 = nn.AvgPool2d((2,2))
        
        


        
        
       
        self.residualconvblock1 = ResidualConvBlock(nIn= 32, nOut=64, timedim=313)
        self.dropout2 = nn.Dropout(0.5)
        self.avgpool2 = nn.AvgPool2d((2,2))
        
        self.residualconvblock2 = ResidualConvBlock(nIn= 64, nOut=128, timedim=156)
        self.dropout3 = nn.Dropout(0.5)
        self.avgpool3 = nn.AvgPool2d((1,2))
        
        self.residualconvblock3 = ResidualConvBlock(nIn= 128, nOut=256, timedim=156)
        self.dropout4 = nn.Dropout(0.5)
        self.avgpool4 = nn.AvgPool2d((1,2))
        
        self.residualconvblock4 = ResidualConvBlock(nIn= 256, nOut=256, timedim=156)
        self.dropout5 = nn.Dropout(0.5)
        self.avgpool5 = nn.AvgPool2d((1,2))
        
        self.residualconvblock5 = ResidualConvBlock(nIn= 256, nOut=256 ,timedim=156)
        self.dropout6 = nn.Dropout(0.5)
        self.avgpool6 = nn.AvgPool2d((1,2))
        
        self.residualconvblock6 = ResidualConvBlock(nIn= 256, nOut=256, timedim=156)
        self.dropout7 = nn.Dropout(0.5)
        self.avgpool7 = nn.AvgPool2d((1,2))
        


    def forward(self, x):
        """
        Forward step of the CNN module
        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)
        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cg1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)



        x = self.residualconvblock1(x)
        x = self.dropout2(x)
        x = self.avgpool2(x)

        x = self.residualconvblock2(x)
        x = self.dropout3(x)
        x = self.avgpool3(x)

        x = self.residualconvblock3(x)
        x = self.dropout4(x)
        x = self.avgpool4(x)

        x = self.residualconvblock4(x)
        x = self.dropout5(x)
        x = self.avgpool5(x)

        x = self.residualconvblock5(x)
        x = self.dropout6(x)
        x = self.avgpool6(x)
        
        x = self.residualconvblock6(x)
        x = self.dropout7(x)
        x = self.avgpool7(x)

        
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

#         self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.cnn = CNN()
        self.rnn = BiGRU(n_in=256, n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

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




