import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pdb

class Encoder(nn.Module):
    def __init__(self)::
        super(cls_model, self).__init__()
        self.layers_common = nn.Sequential(nn.Linear(3, 64),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),)

        self.layers_classifier1 = nn.Sequential(
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 2048),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(2048),)     

    def forward(self, points, young_mod, pois_ratio, force, PoA_x, PoA_y, PoA_z):
        
        B = points.shape[0]
        N = points.shape[1]
        shape = B*N
        points = points.contiguous().cuda()
        points_reshaped = points.view(shape, 3)
        # pdb.set_trace()
        x = self.layers_common(points_reshaped)
        x = self.layers_classifier1(x)
        bottleneck_mu, bottleneck_sig = x[:, 0:1024], x[:, 1024:] 		#split into mu and sigma
        cond_bottleneck_mu = torch.cat((bottleneck_mu, young_mod, pois_ratio, force, PoA_x, PoA_y, PoA_z), dim=1)
        sig_cond = 0.1*torch.ones_like(young_mod)
        cond_bottleneck_sig = torch.cat((bottleneck_sig, sig_cond.repeat(6, dim=1)), dim=1)	#repeat dims?

        return  cond_bottleneck_mu, cond_bottleneck_sig





class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        

    def forward(self, z):
        

class AEModel(nn.Module):
    def __init__(self, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
  

        self.input_shape = input_shape
        self.latent_size = latent_size
        
        self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)

    def forward(self,x):
        x_mean, x_var = self.encoder(x)
        #pdb.set_trace()
        eps = torch.normal(mean = torch.ones_like(x_mean).cuda())

        latent = x_mean+torch.exp(x_var/2)*eps
        x = self.decoder(latent)
        return x, x_mean, x_var
    