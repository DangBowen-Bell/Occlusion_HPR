import torch
import torch.nn as nn
import torch.nn.functional as F


#! GFNet
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

# Input: BxM
# Output: BxN
class ResNetFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResNetFC, self).__init__()

        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

# Input: BxMx3[, BxN]
# Output: BxN
class ResNetPointNet(nn.Module):
    def __init__(self, in_dim=3, h_dim=256, out_dim=256, c_dim=None):
        super(ResNetPointNet, self).__init__()

        if c_dim is not None:
            self.fc_in = nn.Linear(in_dim, h_dim)
        else:
            self.fc_in = nn.Linear(in_dim, 2*h_dim)
        self.block_0 = ResNetFC(2*h_dim, h_dim)
        self.block_1 = ResNetFC(2*h_dim, h_dim)
        self.block_2 = ResNetFC(2*h_dim, h_dim)
        self.block_3 = ResNetFC(2*h_dim, h_dim)
        self.block_4 = ResNetFC(2*h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, out_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool
        
        self.c_dim = c_dim
        self.out_dim = out_dim

    def forward(self, p, c=None):
        net = self.fc_in(p)
        if self.c_dim is not None and c is not None:
            c = c.unsqueeze(1).expand(net.size())
            net = torch.cat([net, c.expand(net.size())], dim=2)

        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        net = self.pool(net, dim=1)

        z = self.fc_out(self.actvn(net))

        return z

# Input: BxN
# Output: BxO
class GFDecoder(nn.Module):
    def __init__(self, out_dim=2, latent_dim=512, 
                 dims=[512, 512, 512, 1024, 512, 512, 512, 512], 
                 dropout_layers=[0, 1, 2, 3, 4, 5, 6, 7], latent_layers=[4],
                 use_norm=True, norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
                 dropout_prob=0.2, latent_dropout=False, xyz_in_all=False):
        super(GFDecoder, self).__init__()

        dims = [latent_dim + 3] + dims + [out_dim]
        self.layer_num = len(dims)
        self.dropout_layers = dropout_layers
        self.latent_layers = latent_layers
        self.use_norm = use_norm
        self.norm_layers = norm_layers
        self.dropout_prob = dropout_prob
        self.latent_dropout = latent_dropout
        self.xyz_in_all = xyz_in_all
        
        for layer in range(0, self.layer_num - 1):
            if layer + 1 in latent_layers:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.layer_num - 2:
                    out_dim -= 3

            if use_norm and layer in self.norm_layers:
                setattr(self, "lin" + str(layer), nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)))
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (not use_norm) and layer in self.norm_layers:
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=self.dropout_prob, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.layer_num - 1):
            if layer in self.latent_layers:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)

            lin = getattr(self, "lin" + str(layer))
            x = lin(x)

            if layer < self.layer_num - 2:
                if (not self.use_norm) and layer in self.norm_layers:
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if layer in self.dropout_layers:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        x = self.tanh(x)

        return x


#! FZNet
# Input: scene (BxMx3), depth (BxNx3), xyz (BxSx3)
# Output: udf (BxSx2)
class FZNet(nn.Module):
    def __init__(self):
        super(FZNet, self).__init__()
        self.encoder_scene = ResNetPointNet(out_dim=256)
        self.encoder_depth = ResNetPointNet(out_dim=256, c_dim=256)
        self.udf_decoder = GFDecoder(out_dim=2)

    def forward(self, x_scene, x_depth, xyz):
        batch_size = xyz.shape[0]
        sample_num = xyz.shape[1]
        
        batch_sample_num = batch_size * sample_num
        xyz = xyz.reshape(batch_sample_num, 3)

        x_scene = self.encoder_scene(x_scene)
        x_depth = self.encoder_depth(x_depth, x_scene)
        
        latent_scene = x_scene.repeat_interleave(sample_num, dim=0)
        latent_depth = x_depth.repeat_interleave(sample_num, dim=0)
    
        decoder_input = torch.cat([latent_scene, latent_depth, xyz], 1)
        udf = self.udf_decoder(decoder_input)
        udf = udf.reshape(batch_size, sample_num, 2)

        return udf
