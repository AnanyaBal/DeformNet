import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import ipdb

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.layers_common = nn.Sequential(nn.Linear(3, 64),
        #                             nn.ReLU(),
        #                             nn.Linear(64, 64),
        #                             nn.ReLU(),)

        # self.layers_classifier1 = nn.Sequential(
        #                             nn.Linear(64, 128),
        #                             nn.ReLU(),
        #                             nn.Linear(128, 1024),
        #                             nn.ReLU(),
        #                             nn.Linear(1024, 2048),
        #                             nn.ReLU(),)  
        self.layers_common = nn.Sequential(nn.Linear(3,64),
                                    nn.ReLU(),
                                    nn.Linear(64,64),
                                    nn.ReLU(),
                                    nn.Linear(64,64),
                                    nn.ReLU(),
                                    nn.Linear(64,128),
                                    nn.ReLU(),
                                    nn.Linear(128,1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,2048),
                                    nn.ReLU(),)


    def forward(self, points, young_mod, pois_ratio, force, PoA_v):

        
        B = points.shape[0]
        N = points.shape[1]
        shape = B*N


        # young_mod  = torch.Tensor([young_mod])
        # pois_ratio = torch.Tensor([pois_ratio])
        # force = torch.Tensor([force])
        # PoA_x = torch.Tensor([PoA_x])
        # PoA_y = torch.Tensor([PoA_y])
        # PoA_z = torch.Tensor([PoA_z])

        points = points.cuda()
        points_reshaped = points.view(shape, 3)
        x = self.layers_common(points_reshaped)
        x = x.view(B,N,-1)
        x = x.amax(dim=1)
        bottleneck_mu, bottleneck_sig = x[:, 0:1024], x[:, 1024:] 		#split into mu and sigma


        conditional_vec_mean = torch.cat((young_mod.unsqueeze(-1), pois_ratio.unsqueeze(-1), 
                                     force.unsqueeze(-1), PoA_v.unsqueeze(-1)),dim=1).cuda()
        # ipdb.set_trace()
        conditional_vec_var = torch.cat((0.1*torch.ones(B,1),0.1*torch.ones(B,1),0.1*torch.ones(B,1),
                                         1*torch.ones(B,1)),dim=1).cuda()

        # ipdb.set_trace()

        cond_bottleneck_mu = torch.cat((bottleneck_mu, conditional_vec_mean),dim=1)
        cond_bottleneck_sig = torch.cat((bottleneck_sig, conditional_vec_var),dim=1)

        # sig_cond = 0.1*torch.ones_like(young_mod)
        # cond_bottleneck_sig = torch.cat((bottleneck_sig, sig_cond.repeat(6, dim=1)), dim=1)	#repeat dims?

        return  cond_bottleneck_mu, cond_bottleneck_sig


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape, num_points=2000):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
        			nn.Linear(latent_dim, 512),
        			nn.ReLU(),
        			nn.Linear(512, num_points*3),
        			nn.Tanh(),
        			)
        self.num_points = num_points

    def forward(self, z):
        # ipdb.set_trace()
        B = z.shape[0]
        out = self.fc(z)
        out = out.view(B,self.num_points,3)
        return out

class MeshVAEModel(nn.Module):

    def __init__(self, latent_size=1028, input_shape = (3, 32, 32)):

        super().__init__()
  

        self.input_shape = input_shape
        self.latent_size = latent_size
        
        self.encoder = Encoder()
        self.decoder = Decoder(latent_size, input_shape)


    def forward(self,x, young_mod, pois_ratio, force, PoA_v):
        x_mean, x_var = self.encoder(x, young_mod, pois_ratio, force, PoA_v)

        eps = torch.normal(mean = torch.ones_like(x_mean).cuda())
        # ipdb.set_trace()
        latent = x_mean+torch.exp(x_var/2)*eps
        x = self.decoder(latent)
        # ipdb.set_trace()
        return x, x_mean, x_var
    