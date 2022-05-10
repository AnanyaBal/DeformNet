import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import ipdb
import os
import pytorch3d
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


    def forward(self, points, young_mod, pois_ratio,force, PoA_v,):

        
        B = points.shape[0]
        N = points.shape[1]
        shape = B*N
        # ipdb.set_trace()

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
        bottleneck_mu, bottleneck_sig = x[:, 0:1024], x[:, 1024:]       #split into mu and sigma


        # ipdb.set_trace()
        conditional_vec_mean = torch.cat((young_mod, pois_ratio, 
                                     force, PoA_v),dim=1).cuda()
        conditional_vec_var = torch.cat((0.1*torch.ones(B,1),0.1*torch.ones(B,1),0.1*torch.ones(B,1),
                                         1*torch.ones(B,1)),dim=1).cuda()

        # ipdb.set_trace()

        cond_bottleneck_mu = torch.cat((bottleneck_mu, conditional_vec_mean),dim=1)
        cond_bottleneck_sig = torch.cat((bottleneck_sig, conditional_vec_var),dim=1)

        # sig_cond = 0.1*torch.ones_like(young_mod)
        # cond_bottleneck_sig = torch.cat((bottleneck_sig, sig_cond.repeat(6, dim=1)), dim=1)   #repeat dims?

        return  cond_bottleneck_mu, cond_bottleneck_sig

class Encoder_PointnetPlusPlus(nn.Module):
    def __init__(self):
        super(Encoder_PointnetPlusPlus, self).__init__()

        self.layers1 = nn.Sequential(nn.Linear(3, 64),
                                nn.ReLU(),
                                # nn.BatchNorm1d(64),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                # nn.BatchNorm1d(64),
                                nn.Linear(64, 128),
                                nn.ReLU(),)
                                # nn.BatchNorm1d(128),)

        self.layers2 = nn.Sequential(nn.Linear(128, 128),
                                nn.ReLU(),
                                # nn.BatchNorm1d(128),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                # nn.BatchNorm1d(128),
                                nn.Linear(128, 256),
                                nn.ReLU(),)
                                # nn.BatchNorm1d(256),)

        self.layers3 = nn.Sequential(nn.Linear(256, 256),
                                nn.ReLU(),
                                # nn.BatchNorm1d(256),
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                # nn.BatchNorm1d(512),
                                nn.Linear(512, 1024),
                                nn.ReLU(),
                                # nn.BatchNorm1d(1024),
                                nn.Linear(1024,2048),
                                nn.ReLU(),)
                                # nn.BatchNorm1d(2048),)

    def index_points(self, points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(self, xyz, npoint):
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


    def forward(self, points, young_mod, pois_ratio,force, PoA_v):
         # ipdb.set_trace()
        B = points.shape[0]
        N = points.shape[1]
        centroids_ind1 = self.farthest_point_sample(points, 512)
        centroids1 = self.index_points(points, centroids_ind1)
        _,points_idx1,_ = pytorch3d.ops.ball_query(centroids1, points, K=50)
        points2 = self.index_points(points, points_idx1)

        d1, d2, d3, d4 = points2.shape
        points2 = points2.reshape(d1*d2*d3, d4)

        out1 = self.layers1(points2)
        out1 = out1.reshape(d1, d2, d3, -1)
        points3 = torch.amax(out1, dim=2)


        centroids_ind2 = self.farthest_point_sample(points3, 128)
        centroids2 = self.index_points(points3, centroids_ind2)
        _,points_idx2,_ = pytorch3d.ops.ball_query(centroids2, points3, K=50, radius=0.4)
        points3 = self.index_points(points3, points_idx2)

        d1, d2, d3, d4 = points3.shape
        points3 = points3.reshape(d1*d2*d3, d4)
        # pdb.set_trace()
        out2 = self.layers2(points3)
        out2 = out2.reshape(d1, d2, d3, -1)
        points4 = torch.amax(out2, dim=2)


        d1, d2, d3 = points4.shape
        points4 = points4.reshape(d1*d2, d3)
        points5 = self.layers3(points4)

        points5 = points5.reshape(d1, d2, -1)

        x = torch.amax(points5, dim=1)
        # ipdb.set_trace()
        
        bottleneck_mu, bottleneck_sig = x[:, 0:1024], x[:, 1024:]       #split into mu and sigma
        conditional_vec_mean = torch.cat((young_mod, pois_ratio, 
                                     force, PoA_v),dim=1).cuda()
        conditional_vec_var = torch.cat((0.1*torch.ones(B,1),0.1*torch.ones(B,1),0.1*torch.ones(B,1),
                                         1*torch.ones(B,1)),dim=1).cuda()


        cond_bottleneck_mu = torch.cat((bottleneck_mu, conditional_vec_mean),dim=1)
        cond_bottleneck_sig = torch.cat((bottleneck_sig, conditional_vec_var),dim=1)

        return  cond_bottleneck_mu, cond_bottleneck_sig


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape, num_points=5000):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(1028, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, num_points*3),
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

    def __init__(self, latent_size=1028, input_shape = (3, 32, 32),pointnet=True):

        super().__init__()
  

        self.input_shape = input_shape
        self.latent_size = latent_size
        pointnet=False
        if pointnet:
            self.encoder = Encoder()
            print("Loading PointNet")
        else:
            self.encoder = Encoder_PointnetPlusPlus()
            print("Loading PointNet++")
        self.decoder = Decoder(latent_size, input_shape)


    def forward(self,x, young_mod, pois_ratio, force, PoA_v):
        x_mean, x_var = self.encoder(x, young_mod, pois_ratio, force, PoA_v)

        eps = torch.normal(mean = torch.ones_like(x_mean).cuda())
        # ipdb.set_trace()
        latent = x_mean+torch.exp(x_var/2)*eps
        x = self.decoder(latent)
        # ipdb.set_trace()
        return x, x_mean, x_var
    
class DeformNet(nn.Module):

    def __init__(self, cycle_consistency=False,pointnet=True):

        super().__init__()

        self.cycle_consistency = cycle_consistency
        self.meshvae = MeshVAEModel(pointnet)

    def forward(self,x, young_mod, pois_ratio, force, PoA_v):

        if self.cycle_consistency:
            # print("Loading cycle_consistency")
            x_p, x_mean_p, x_var_p = self.meshvae(x, young_mod, pois_ratio, force, PoA_v)
            x_recon,x_mean_recon,x_var_recon = self.meshvae(x_p, young_mod, pois_ratio, -force, PoA_v)

            return x_recon,x_mean_recon,x_var_recon,x_p,x_mean_p,x_var_p
        else:
            x_p, x_mean_p, x_var_p = self.meshvae(x, young_mod, pois_ratio, force, PoA_v)
            return x_p,x_mean_p,x_var_p