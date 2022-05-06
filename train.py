import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import MeshVAEModel, DeformNet
# from data_loader import get_data_loader
# from utils import save_checkpoint, create_dir
import ipdb

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
import glob
from mesh_loader import *
from losses import *
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
save_path = "./model/"

import csv
import os 


# def read_conditioning_params(idx):
#     idx = [ i+2 for i in idx]
#     file = open('DeformNet Data Details - Ansys - Sheet2.csv')
#     print('reading CSV')
#     csvreader = csv.reader(file)
#     counter = 0
#     rows = []

#     cond_params_batch = []

#     for row in csvreader:
#         ipdb.set_trace()   
#         if (row[3]!=''):
#             counter+=1
#             if (counter in idx):
#                 # print(counter)
#                 return_list = ([row[4], row[3], row[5], row[9], row[10], row[11]])
#                 return_list = [float(item) for item in return_list]              
#                 cond_params_batch.append(return_list) 
#     # ipdb.set_trace()

#     return torch.tensor(cond_params_batch)

def read_conditioning_params(idx,data_path):

    file = open(data_path+'DeformNet Data Details - Ansys - Sheet2.csv')
    print('reading CSV')
    csvreader = csv.reader(file)
    counter = 0
    pr = []
    ym =[]
    for idx,row in enumerate(csvreader):
        if idx>1:
            pr += [float(row[3])]*10
            ym += [float(row[4])]*10

    pr = torch.from_numpy(np.array(pr)).unsqueeze(-1)
    ym = torch.from_numpy(np.array(ym)).unsqueeze(-1)

    # ipdb.set_trace()

    return ym,pr
def get_dataset(data_path):

    sample_points = 2000
    deformed_file = data_path + 'deformed/*.json'
    undeformed_file = data_path + 'undeformed/cube_0.json'

    # read undeformed list
    file = open(undeformed_file)
    data = json.load(file)
    verts_undeformed_list = []
    x = np.array(data[0]['x_coord'])
    y = np.array(data[0]['y_coord'])
    z = np.array(data[0]['z_coord'])-0.1
    force_applied = data[0]['force']
    x_poa = data[0]['x_value_force']
    y_poa = data[0]['y_value_force']
    z_poa = data[0]['z_value_force']-0.1
    # ipdb.set_trace()
    verts_undeformed_list.append(np.vstack((x,y,z)).T)
    verts_undeformed_list = torch.from_numpy(np.array(verts_undeformed_list)).cuda()
    verts_undeformed_list = normalize_pc(verts_undeformed_list)
    indices = torch.tensor(random.sample(range(verts_undeformed_list.shape[1]), sample_points))
    verts_undeformed_list = verts_undeformed_list[:,indices,:]

    verts_deformed_list = []
    poa_list = []
    force_applied_list = []

    #read deformed file
    for idx,f in enumerate(sorted(glob.glob(deformed_file))):

        file = open(f)
        data = json.load(file)

        for l in range(len(data)):
            x = np.array(data[l]['x_coord'])
            y = np.array(data[l]['y_coord'])
            z = np.array(data[l]['z_coord'])-0.1
            force_applied = data[l]['force']
            x_poa = data[l]['x_value_force']
            y_poa = data[l]['y_value_force']
            z_poa = data[l]['z_value_force']-0.1
            # ipdb.set_trace()
            verts_deformed_list.append(np.vstack((x,y,z)).T)
            poa_list.append(np.array([x_poa,y_poa,z_poa]).T)
            force_applied_list.append(force_applied)
        ym,pr = read_conditioning_params(idx,data_path)

    verts_deformed_list = torch.from_numpy(np.array(verts_deformed_list)).cuda()
    poa_list = torch.from_numpy(np.array(poa_list)).cuda()
    force_applied_list = torch.from_numpy(np.array(force_applied_list)).unsqueeze(-1)
    verts_deformed_list = normalize_pc(verts_deformed_list)


    verts_deformed_list = verts_deformed_list[:,indices,:]
    verts_undeformed_list = verts_undeformed_list.repeat(verts_deformed_list.shape[0],1,1)
    poi = find_closest_vertex(verts_undeformed_list,poa_list)

    idx_it = [i for i in range(10)]

    # ipdb.set_trace()
    # ipdb.set_trace()
    return verts_undeformed_list,verts_deformed_list, force_applied_list, poi, ym, pr
        # ipdb.set_trace()

# def get_dataset(data_path):

#     deformed_path = data_path+'deformed/*.obj'
#     undeformed_path = data_path+'undeformed/*.obj'

#     verts_list = []
#     faces_list = []

#     for f in sorted(glob.glob(undeformed_path)):

#         verts,faces = load_mesh(f)
#         verts_list.append(verts)
#         faces_list.append(faces)
#     mesh = get_mesh(verts_list,faces_list)
#     verts_undeformed    = pytorch3d.ops.sample_points_from_meshes(mesh,num_samples=2000)

#     # verts_undeformed = torch.Tensor(verts_undeformed)
#     verts_list =  []
#     faces_list =  []
#     cond_vector = []
#     for idx,f in enumerate(sorted(glob.glob(deformed_path))):

#         verts,faces = load_mesh(f)
#         verts_list.append(verts)
#         faces_list.append(faces)
#         # cond_vector.append(read_csv(idx))

#     mesh = get_mesh(verts_list,faces_list)
#     verts_deformed    = pytorch3d.ops.sample_points_from_meshes(mesh,num_samples=2000)

#     verts_undeformed = verts_undeformed.repeat(verts_deformed.shape[0],1,1)
#     # verts_deformed = torch.Tensor(verts_deformed)


#     return verts_undeformed,verts_deformed

def find_closest_vertex(verts_undeformed, pos):
    closest_vertices = []
    p1 = pos.view(pos.shape[0],1,3)
    idx = pytorch3d.ops.knn_points(p1,verts_undeformed,K=1)[1].squeeze(-1)
    # ipdb.set_trace()

    # distances_deformed = ((verts_deformed - pos)**2).sum(1)**0.5
    # closes_vertex_deformed = distances_deformed.argmin()

    return  idx            #closest_vertex_undeformed, closes_vertex_deformed


# def train(train_dataloader, model, opt, epoch, args, writer):
def train(model,writer,cycle_consistency):
    
    # ipdb.set_trace()
    model.train()
    batch_size = 20
    # step = epoch*len(train_dataloader)
    epoch_loss = 0
    viz_data = True
    opt = optim.Adam(model.parameters(), 1e-6)
    # ipdb.set_trace()
    verts_undeformed_total,verts_deformed_total,force_total,poi_total,ym_total,pr_total = get_dataset(args.data_path)
    print("Total data  points: {}".format(verts_deformed_total.shape[0]))
    count = 0
    epoch_loss_list = []

    c1 = torch.tensor([[1.,1.,0]]).unsqueeze(0).repeat(1,2000,1)
    c2 = torch.tensor([[1.,0,1.]]).unsqueeze(0).repeat(1,2000,1)
    c3 = torch.tensor([[0,1.,1.]]).unsqueeze(0).repeat(1,2000,1)
    color_all = torch.cat((c1,c3),dim=1)
    idx_it = [i for i in range(24)]

    for epoch in range(1000):
        epoch_loss = 0
        for b_idx,b in enumerate(range(verts_deformed_total.shape[0]//batch_size)):

            # print(b_idx)

            verts_deformed = verts_deformed_total[b_idx*batch_size:(b_idx+1)*batch_size]
            verts_undeformed = verts_undeformed_total[b_idx*batch_size:(b_idx+1)*batch_size]
            force = force_total[b_idx*batch_size:(b_idx+1)*batch_size]
            poi = poi_total[b_idx*batch_size:(b_idx+1)*batch_size]
            ym = ym_total[b_idx*batch_size:(b_idx+1)*batch_size]
            pr = pr_total[b_idx*batch_size:(b_idx+1)*batch_size]

            # ipdb.set_trace()
            # verts_undeformed, verts_deformed =verts_deformed_total, verts_undeformed_total

            verts_undeformed = verts_undeformed.float().to(args.device)
            verts_deformed = verts_deformed.float().to(args.device)
            force = force.float().to(args.device)
            poi = poi.float().to(args.device)
            ym = ym.float().to(args.device)
            pr = pr.float().to(args.device)

            if cycle_consistency:
                predictions_recon,_,_,predictions,_,_ = model(verts_undeformed, force, poi, ym, pr)
                chamfer_loss_recon = chamfer_loss(predictions_recon,verts_undeformed)
                chamfer_loss_pred = chamfer_loss(predictions,verts_deformed)
                chamfer_loss_ = (chamfer_loss_recon + chamfer_loss_pred)/2
            else:
                predictions,latent_mean,latent_var = model(verts_undeformed, force, poi, ym, pr)
                chamfer_loss_ = chamfer_loss(predictions,verts_deformed)

            if viz_data and epoch%100==0 and b_idx == 0 :
                # xyz = verts_deformed[0,int(poi[0]),:]
                # point = verts_deformed[poi]
                # z_arrow = torch.linspace(1,1.5,50).reshape(-1,1).cuda()
                # x_arrow = torch.tensor(xyz[0]).reshape(-1,1).repeat(50,1).cuda()
                # y_arrow = torch.tensor(xyz[1]).reshape(-1,1).repeat(50,1).cuda()
                # # ipdb.set_trace()
                # arrow_coord = torch.cat((x_arrow,y_arrow,z_arrow),dim=-1).unsqueeze(0)
                # for i in range(10):
                # #     # ipdb.set_trace()
                #     viz_pointcloud(verts_deformed[i],'pc_deformed_{}'.format(epoch))
                # viz_pointcloud(verts_undeformed[5],'pc_undeformed_{}'.format(epoch))
                # viz_pointcloud(predictions[5].detach(),'pc_predicted_{}'.format(epoch))

                # # test(verts_undeformed[i],600,)
                
                pc_total = torch.cat((verts_deformed[5],predictions[5].detach()),dim=0)
                viz_pointcloud(pc_total,'pc_combined_{}'.format(b_idx),color_all)
# 

            # Compute Loss
            # kl_loss_ = kl_loss(latent_mean,latent_var)
            # ipdb.set_trace()
            loss_total = chamfer_loss_#+kl_loss_
            epoch_loss += loss_total
            # Backward and Optimize
            opt.zero_grad()
            loss_total.backward()
            opt.step()
        count += 1
        print("Loss at epoch {} and batch {} is {} ".format(epoch,b_idx,epoch_loss/batch_size))
        writer.add_scalar("train/loss",epoch_loss/batch_size,epoch)
        epoch_loss_list.append(epoch_loss.item()/batch_size)

        if epoch%100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                }, save_path+'last.pt')
        # writer.add_scalar('train_loss', loss.item(), step+i)
        # print("loss total: ",epoch_loss)
    count_idx = np.arange(0,count)
    #ipdb.set_trace()
    plt.plot(count_idx,epoch_loss_list)
    plt.show()

    return epoch_loss

def test(verts_undeformed,poi,force=600,ym=0.001,pr=0.47):
    
    model.eval()

    # Evaluation in Classification Task
    force = torch.tensor(force).to(args.device)
    poi = torch.tensor(poi).to(args.device)
    ym = torch.tensor(ym).to(args.device)
    pr = torch.tensor(pr).to(args.device)

    predictions,_,_ = model(verts_undeformed, force, poi, ym, pr)

   
    return accuracy

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--data_path', type=str, default='/home/cobra/abhimanyu_course/deformation/data/', help='root folder for data')


    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10)
    parser.add_argument('--default_cond_params', type=str, default='no')
    parser.add_argument('--load_checkpoint', type=str, default='')
    

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    writer = SummaryWriter()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific
    # ipdb.set_trace()
    # undeformed_verts, deformed_verts = get_dataset(args.data_path)
    # model = MeshVAEModel().cuda()
    cycle_consistency = True
    model = DeformNet(cycle_consistency=cycle_consistency).cuda()
    train(model,writer,cycle_consistency)
    # ipdb.set_trace()

