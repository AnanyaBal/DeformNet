import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import MeshVAEModel
# from data_loader import get_data_loader
# from utils import save_checkpoint, create_dir
import ipdb

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
import glob
from mesh_loader import *
from losses import *
from torch.utils.tensorboard import SummaryWriter

import csv
import os 


def read_conditioning_params(idx):
    idx = [ i+2 for i in idx]
    file = open('DeformNet Data Details - Ansys - Sheet1.csv')
    print('reading CSV')
    # ipdb.set_trace()   
    csvreader = csv.reader(file)
    counter = 0
    rows = []

    cond_params_batch = []

    for row in csvreader:
        if (row[3]!=''):
            counter+=1
            if (counter in idx):
                # print(counter)
                return_list = ([row[4], row[3], row[5], row[9], row[10], row[11]])
                return_list = [float(item) for item in return_list]              
                cond_params_batch.append(return_list) 
    # ipdb.set_trace()

    return torch.tensor(cond_params_batch)


def get_dataset(data_path):

    deformed_path = data_path+'deformed/*.obj'
    undeformed_path = data_path+'undeformed/*.obj'

    verts_list = []
    faces_list = []

    for f in sorted(glob.glob(undeformed_path)):

        verts,faces = load_mesh(f)
        verts_list.append(verts)
        faces_list.append(faces)
    mesh = get_mesh(verts_list,faces_list)
    verts_undeformed    = pytorch3d.ops.sample_points_from_meshes(mesh,num_samples=2000)

    # verts_undeformed = torch.Tensor(verts_undeformed)
    verts_list =  []
    faces_list =  []
    cond_vector = []
    for idx,f in enumerate(sorted(glob.glob(deformed_path))):

        verts,faces = load_mesh(f)
        verts_list.append(verts)
        faces_list.append(faces)
        # cond_vector.append(read_csv(idx))

    mesh = get_mesh(verts_list,faces_list)
    verts_deformed    = pytorch3d.ops.sample_points_from_meshes(mesh,num_samples=2000)

    verts_undeformed = verts_undeformed.repeat(verts_deformed.shape[0],1,1)
    # verts_deformed = torch.Tensor(verts_deformed)


    return verts_undeformed,verts_deformed

def find_closest_vertex(verts_undeformed, verts_deformed, x, y, z):
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    pos = torch.hstack((x,y,z))
    # point_tensor = pos.repeat(verts_undeformed.shape[0])
    # ipdb.set_trace()
    closest_vertices = []
    for i in range(verts_undeformed.shape[0]):
        rd = verts_undeformed.shape[1]
        xyz = pos[i,:].unsqueeze(-1).repeat(rd, 1).reshape(rd,3).cuda()
        distance_undeformed = ((verts_undeformed[i,:,:] - xyz)**2).sum(1)**0.5
        closest_vertex_undeformed = distance_undeformed.argmin()
        closest_vertices.append(closest_vertex_undeformed)

    # ipdb.set_trace()
    # distances_deformed = ((verts_deformed - pos)**2).sum(1)**0.5
    # closes_vertex_deformed = distances_deformed.argmin()

    return  closest_vertices            #closest_vertex_undeformed, closes_vertex_deformed


# def train(train_dataloader, model, opt, epoch, args, writer):
def train(model):
    
    # ipdb.set_trace()
    model.train()
    # step = epoch*len(train_dataloader)
    epoch_loss = 0
    viz_data = True
    opt = optim.Adam(model.parameters(), 0.001, betas=(0.9, 0.999))
    # ipdb.set_trace()
    verts_undeformed,verts_deformed = get_dataset(args.data_path)
    count = 0
    epoch_loss_list = []

    c1 = torch.tensor([[1.,1.,0]]).unsqueeze(0).repeat(1,2000,1)
    c2 = torch.tensor([[1.,0,1.]]).unsqueeze(0).repeat(1,2000,1)
    c3 = torch.tensor([[0,1.,1.]]).unsqueeze(0).repeat(1,2000,1)
    color_all = torch.cat((c1,c3),dim=1)
    idx_it = [i for i in range(24)]

    for epoch in range(100):
        print("Epoch: ",epoch)
        epoch_loss = 0
        for i, batch in enumerate(range(verts_deformed.shape[0])):
            # ipdb.set_trace()
            # verts_undeformed, verts_deformed =verts_deformed_total, verts_undeformed_total

            verts_undeformed = verts_undeformed.to(args.device)
            verts_deformed = verts_deformed.to(args.device)

            cond_params = read_conditioning_params((i+1)*idx_it)
            # ipdb.set_trace()

            closest_vertices_undeformed = torch.tensor(find_closest_vertex(verts_undeformed, verts_deformed, 
                                                 cond_params[:,3], cond_params[:,4], cond_params[:,5]))

            # ------ TO DO: Forward Pass ------
            # predictions,latent_mean,latent_var = model(verts_undeformed, cond_params[:,0], 
            #                                     cond_params[:,1], cond_params[:,2],  cond_params[:,3],
            #                                     cond_params[:,4], cond_params[:,5])

            predictions,latent_mean,latent_var = model(verts_undeformed, cond_params[:,0], cond_params[:,1], 
                                                        cond_params[:,2], closest_vertices_undeformed)
            if viz_data and i==0 and epoch%10==0:
                # viz_pointcloud(verts_deformed[20],'pc_deformed_{}'.format(epoch))
                # viz_pointcloud(verts_undeformed[20],'pc_undeformed_{}'.format(epoch))
                # viz_pointcloud(predictions[20].detach(),'pc_predicted_{}'.format(epoch))
                
                pc_total = torch.cat((verts_deformed[20],predictions[20].detach()),dim=0)
                viz_pointcloud(pc_total,'pc_combined_{}'.format(epoch),color_all)


            # Compute Loss
            chamfer_loss_ = chamfer_loss(predictions,verts_deformed)
            kl_loss_ = kl_loss(latent_mean,latent_var)
            # ipdb.set_trace()
            loss_total = chamfer_loss_#+kl_loss_
            epoch_loss += loss_total
            epoch_loss_list.append(loss_total.item())
            # Backward and Optimize
            opt.zero_grad()
            loss_total.backward()
            opt.step()
            count += 1

        # writer.add_scalar('train_loss', loss.item(), step+i)
        print("loss total: ",epoch_loss)
    count_idx = np.arange(0,count)
    #ipdb.set_trace()
    plt.plot(count_idx,epoch_loss_list)
    plt.show()

    return epoch_loss

def test(test_dataloader, model, epoch, args, writer):
    
    model.eval()

    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                pred_labels = torch.argmax(model(point_clouds),dim=1)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj
                
        
    # Evaluation in Segmentation Task
    else:
        correct_point = 0
        num_point = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():     
                pred_labels = torch.argmax(model(point_clouds), dim=2)  #dimension changes to 2 as the output is now B, N, C instead of classificationm output B, C

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_point / num_point

    writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{0}'.format(args.task+"_"+args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        # model = cls_model().cuda()
        model = pt_net_model().cuda()
    else:
        model = seg_model().cuda()
    
    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Dataloader for Training & Testing
    train_dataloader = get_data_loader(args=args, train=True)
    test_dataloader = get_data_loader(args=args, train=False)

    #============================Addition===============================
    mesh_src = ico_sphere(4,'cuda')
    # verts = 
    mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
    #============================Addition End===============================

    print ("successfully loaded data")

    best_acc = -1

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        
        # Test
        current_acc = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print ("======== training completes ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--data_path', type=str, default='/home/ananya/DeformNet-master/data/', help='root folder for data')


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

    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific
    # ipdb.set_trace()
    undeformed_verts, deformed_verts = get_dataset(args.data_path)
    model = MeshVAEModel().cuda()
    train(model)
    ipdb.set_trace()

