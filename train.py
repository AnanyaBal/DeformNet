import numpy as np
import argparse
import torch
import torch.optim as optim
from distutils.version import LooseVersion
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

import csv
import os 
import tqdm
import copy
from pytorch3d.io import load_obj, load_ply


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def read_conditioning_params():

    file = open('DeformNet Data Details - Ansys - Sheet1.csv')
    # print('reading CSV')
    # ipdb.set_trace()   
    csvreader = csv.reader(file)
    counter = 1
    rows = []

    cond_params_batch = []

    for id,row in enumerate(csvreader):
        
        # ipdb.set_trace()
        if (row[5]!='' and id!=0 and id!=1):
            # print(counter)
            # ipdb.set_trace()
            return_list = ([row[4], row[3], row[5], row[9], row[10], row[11]])
            return_list = [float(item) for item in return_list]              
            cond_params_batch.append(return_list)
            # print("file name: {},{} ".format(row[0],counter))
            counter += 1
    # ipdb.set_trace()

    return torch.tensor(cond_params_batch)

def get_dataset(data_path,shapes=['cube'],):

    test_idx = []
    total_idx = [i for i in range(432)]
    for ti in range(18):
        test_idx += [5+(24*ti),11+(24*ti),19+(24*ti),23+(24*ti)]
    train_idx  = copy.deepcopy(total_idx)

    for item in train_idx:
        if item in test_idx:
            train_idx.remove(item)
    # ipdb.set_trace()
    data_aug = True
    if shapes is None:
        shapes = ['prism','cube','cyl']

    deformed_list =[]
    undeformed_list = []
    len_shape_list = []
    for shape in shapes:
        verts_undeformed,verts_deformed,len_shape = get_meshes(args.data_path,shape)
        deformed_list.append(verts_undeformed)
        undeformed_list.append(verts_deformed)
        len_shape_list.append(len_shape)
    if len(deformed_list) == 3:
        verts_undeformed = torch.cat(( undeformed_list[0],undeformed_list[1],undeformed_list[2]),dim=0)
        verts_deformed = torch.cat((deformed_list[0],deformed_list[1],deformed_list[2]),dim=0)

    # ipdb.set_trace()
    cond_params = read_conditioning_params()[:verts_deformed.shape[0]]

    verts_deformed,_,_ = normalize_mesh(verts_deformed)
    verts_undeformed,center,scale = normalize_mesh(verts_undeformed)
    poa = find_closest_vertex(verts_undeformed,normalize_mesh(cond_params[:,3:6].unsqueeze(1),center,scale)[0])
    ym = cond_params[:,0].unsqueeze(-1)
    pr = cond_params[:,1].unsqueeze(-1)
    force = cond_params[:,2].unsqueeze(-1)

    if data_aug:
        ym = torch.cat((ym,ym),dim=0)
        pr = torch.cat((pr,pr),dim=0)
        force = torch.cat((force,force),dim=0)
        poa = torch.cat((poa,poa),dim=0)

        verts_undeformed_rot_cube_cyl = rotate_pc(verts_undeformed[:145],0,1.57,0)
        verts_deformed_rot_cube_cyl = rotate_pc(verts_deformed[:145],0,1.57,0)

        verts_undeformed_rot_prism = rotate_pc(verts_undeformed[145:],0,2.0944,0)
        verts_deformed_rot_prism = rotate_pc(verts_deformed[145:],0,2.0944,0)

        verts_undeformed = torch.cat((verts_undeformed.cuda(),verts_undeformed_rot_cube_cyl.cuda(),verts_undeformed_rot_prism.cuda()),dim=0)
        verts_deformed = torch.cat((verts_deformed.cuda(),verts_deformed_rot_cube_cyl.cuda(),verts_deformed_rot_prism.cuda()),dim=0)

        # ipdb.set_trace()

    dataset = {}
    dataset['train'] = [verts_undeformed[train_idx],verts_deformed[train_idx],ym[train_idx],pr[train_idx],force[train_idx],poa[train_idx]]
    dataset['test'] = [verts_undeformed[test_idx],verts_deformed[test_idx],ym[test_idx],pr[test_idx],force[test_idx],poa[test_idx]]

    return dataset


def viz_meshes():

    center = None
    scale = None
    verts_list = []
    faces_list = []
    path = 'cube_222_sw.obj'
    verts, faces, _ = load_obj(path)
    faces = faces.verts_idx
    verts_list.append(verts)
    faces_list.append(faces)
    mesh = get_mesh(verts_list,faces_list)
    verts    = pytorch3d.ops.sample_points_from_meshes(mesh,num_samples=2000)

    # ipdb.set_trace()
    if center is None:
        center = verts.mean(1)
    # ipdb.set_trace()
    verts = verts - center.unsqueeze(1).repeat(1,verts.shape[1],1)
    if scale is None:
        scale = torch.max(verts.abs().max(1)[0],dim=1)[0].reshape(-1,1,1)
    verts = verts / scale.repeat(1,verts.shape[1],verts.shape[2])
    c1 = torch.tensor([[1.,1.,0]]).unsqueeze(0).repeat(1,2000,1)

    viz_pointcloud(verts[0],'pc_undeformed_cyl',c1)

    


def get_meshes(data_path,shape,):

    deformed_path = data_path+'deformed/{}*.obj'.format(shape)
    undeformed_path = data_path+'undeformed/{}*.obj'.format(shape)
    

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

    # ipdb.set_trace()
    verts_undeformed = verts_undeformed.repeat(verts_deformed.shape[0],1,1)


    # verts_deformed = torch.Tensor(verts_deformed)


    return verts_undeformed,verts_deformed,verts_deformed.shape[0]

def find_closest_vertex(verts_undeformed, pos):
    closest_vertices = []
    p1 = pos.view(pos.shape[0],1,3)
    idx = pytorch3d.ops.knn_points(p1,verts_undeformed,K=1)[1].squeeze(-1)

    return  idx            #closest_vertex_undeformed, closes_vertex_deformed


# def train(train_dataloader, model, opt, epoch, args, writer):
def train(model,writer,cycle_consistency):
    
    # ipdb.set_trace()
    model.train()
    # step = epoch*len(train_dataloader)
    epoch_loss = 0
    viz_data = True
    opt = optim.Adam(model.parameters(), 5e-5, betas=(0.9, 0.999))
    # verts_undeformed_total,verts_deformed_total,ym_total,pr_total,force_total,poa_total = get_dataset(args.data_path,['cube','cyl','prism'],)
    dataset = get_dataset(args.data_path,['cube','cyl','prism'],)
 
    # verts_undeformed_total = verts_undeformed_total.float().cuda()
    # verts_deformed_total = verts_deformed_total.float().cuda()
    # force_total = force_total.float().cuda()
    # poa_total = poa_total.float().cuda()
    # ym_total = ym_total.float().cuda()
    # pr_total = pr_total.float().cuda()

    verts_undeformed_total = dataset['train'][0].float().cuda()
    verts_deformed_total = dataset['train'][1].float().cuda()
    ym_total = dataset['train'][2].float().cuda()
    pr_total = dataset['train'][3].float().cuda()
    force_total = dataset['train'][4].float().cuda()
    poa_total = dataset['train'][5].float().cuda()
    # ipdb.set_trace()

    # for i in range(verts_deformed_total.shape[0]):
    #     viz_pointcloud(verts_deformed_total[i],'pc_deformed_{}'.format(i))

    # ipdb.set_trace()
    count = 0
    epoch_loss_list = []

    c1 = torch.tensor([[1.,1.,0]]).unsqueeze(0).repeat(1,2000,1)
    c2 = torch.tensor([[1.,0,1.]]).unsqueeze(0).repeat(1,2000,1)
    c3 = torch.tensor([[0,1.,1.]]).unsqueeze(0).repeat(1,2000,1)
    color_all = torch.cat((c1,c3),dim=1)

    batch_size = 16

    for epoch in range(300):
        print("Epoch: ",epoch)
        epoch_loss = 0
        for b_idx,b in tqdm.tqdm(enumerate(range(verts_deformed_total.shape[0]//batch_size))):

            verts_deformed = verts_deformed_total[b_idx*batch_size:(b_idx+1)*batch_size]
            verts_undeformed = verts_undeformed_total[b_idx*batch_size:(b_idx+1)*batch_size]
            force = force_total[b_idx*batch_size:(b_idx+1)*batch_size]
            poa = poa_total[b_idx*batch_size:(b_idx+1)*batch_size]
            ym = ym_total[b_idx*batch_size:(b_idx+1)*batch_size]
            pr = pr_total[b_idx*batch_size:(b_idx+1)*batch_size]
        # ipdb.set_trace()
        # verts_undeformed, verts_deformed =verts_deformed_total, verts_undeformed_total

            cycle_consistency = cycle_consistency


            if cycle_consistency:
                predictions_recon,_,_,predictions,_,_ = model(verts_undeformed, ym,pr,force,poa)
                chamfer_loss_recon = chamfer_loss(predictions_recon,verts_undeformed)
                chamfer_loss_pred = chamfer_loss(predictions,verts_deformed)
                chamfer_loss_ = (chamfer_loss_recon + chamfer_loss_pred)/2
            else:
                predictions,latent_mean,latent_var = model(verts_undeformed, ym,pr,force,poa)
                chamfer_loss_ = chamfer_loss(predictions,verts_deformed)

            # ipdb.set_trace()

            if viz_data and b_idx==6 and epoch%10==0:
                pass
                # viz_pointcloud(verts_deformed[10],'pc_deformed_{}'.format(epoch))
                viz_pointcloud(verts_deformed[11],'pc_deformed_cyl_1'.format(epoch),c1)
                viz_pointcloud(verts_undeformed[11],'pc_undeformed_prism'.format(epoch),c1)
                ipdb.set_trace()
                # viz_pointcloud(verts_deformed[12],'pc_deformed_prism'.format(epoch))
                # viz_pointcloud(predictions[10].detach(),'pc_predicted_{}'.format(epoch))
                
                # pc_total = torch.cat((verts_deformed[20],predictions[20].detach()),dim=0)
                # viz_pointcloud(pc_total,'pc_combined_{}'.format(epoch),color_all)


            # Compute Loss
            # chamfer_loss_ = chamfer_loss(predictions,verts_deformed)
            # kl_loss_ = kl_loss(latent_mean,latent_var)
            loss_total = chamfer_loss_#+kl_loss_
            epoch_loss += loss_total
            epoch_loss_list.append(loss_total.item())
            # Backward and Optimize
            opt.zero_grad()
            loss_total.backward()
            opt.step()
        count += 1

        # writer.add_scalar('train_loss', loss.item(), step+i)
        print("Loss at epoch {} and batch {} is {} ".format(epoch,b_idx,epoch_loss/batch_size))

        writer.add_scalar("train/loss",epoch_loss/batch_size,epoch)

        if epoch%10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                },'./pn++/vae_cc_last.pt')

            # test(model,dataset)


    return epoch_loss

def test(model,dataset = None,poi=1000,force=1000,ym=0.001,pr=0.47,x=0.,y=0.,z=0.1):
    
    pred = []
    shape_ = 'prism'
    # verts_undeformed_total,verts_deformed_total,ym_total,pr_total,force_total,poa_total = get_dataset(args.data_path,[shape_])
 
    # verts_undeformed_total = verts_undeformed_total.float().cuda()
    # verts_deformed_total = verts_deformed_total.float().cuda()
    # force_total = force_total.float().cuda()
    # poa_total = poa_total.float().cuda()
    # ym_total = ym_total.float().cuda()
    # pr_total = pr_total.float().cuda()

    # verts_undeformed = verts_undeformed_total[0].unsqueeze(0)
    # verts_deformed   = verts_deformed_total[0].unsqueeze(0)
    # ym               = ym_total[0].unsqueeze(0)
    # poa              = poa_total[0].unsqueeze(0)
    # pr               = pr_total[0].unsqueeze(0)
    # # force            = force_total[0].unsqueeze(0)
    # force            = torch.tensor(0000).reshape(1,1).float().cuda()

    verts_undeformed_total = dataset['test'][0].float().cuda()
    verts_deformed_total = dataset['test'][1].float().cuda()
    ym_total = dataset['test'][2].float().cuda()
    pr_total = dataset['test'][3].float().cuda()
    force_total = dataset['test'][4].float().cuda()
    poa_total = dataset['test'][5].float().cuda()
    
    for i in range(9):
        predictions,latent_mean,latent_var = model(verts_undeformed_total[4*i:4*(i+1)], ym_total[4*i:4*(i+1)],pr_total[4*i:4*(i+1)],force_total[4*i:4*(i+1)],poa_total[4*i:4*(i+1)])
        # ipdb.set_trace()
        chamfer_loss_ = chamfer_loss(predictions,verts_deformed_total[4*i:4*(i+1)])
        print("chamfer_loss is: {}".format(chamfer_loss_))


    print("pred loss test: ")

        # ipdb.set_trace()
        # poi = find_closest_vertex(verts_undeformed, torch.tensor([0.,0.1,0.1]).unsqueeze(0).cuda())

        # Evaluation in Classification Task
        # ipdb.set_trace()

        # pred.append(predictions[0].detach())

    # viz_pointcloud(predictions[0].detach(),'pc_predicted_test_{}'.format(force))
    # viz_pointcloud(verts_undeformed[0],'pc_undeformed')
    # c1 = torch.tensor([[1.,0.,0]]).unsqueeze(0).repeat(1,5000,1)
    # c2 = torch.tensor([[0.,1.,0.]]).unsqueeze(0).repeat(1,5000,1)
    # c3 = torch.tensor([[0.,0.,1.]]).unsqueeze(0).repeat(1,5000,1)
    # ipdb.set_trace()
    # color_all = torch.cat((c1,c3),dim=1)
    # pc_total = torch.cat((pred[0],pred[2]),dim=0)
    # viz_pointcloud(pred[0].detach(),'pc_test_{}_{}_0'.format(0,shape_),c2)
    # viz_pointcloud(pred[1],'pc_test_{}'.format(1),c2)
    # viz_pointcloud(pred[2],'pc_test_{}'.format(2),c2)

    # print("Chamfer loss: ",chamfer_loss(predictions,verts_undeformed))
    # np.savez('pc_predicted_test_{}'.format(int(force)),verts_deformed[0].cpu().numpy())


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
    parser.add_argument('--data_path', type=str, default='/home/cobra/abhimanyu_course/DeformNet/data/', help='root folder for data')


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
    # viz_meshes()
    parser = create_parser()
    args = parser.parse_args()
    writer = SummaryWriter()

    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific
    # ipdb.set_trace()
    # undeformed_verts, deformed_verts = get_dataset(args.data_path)

    # train(model)
    # ipdb.set_trace()
    is_test = False
    pointnet = True
    # model_name = 'vae_cc_last.pt'
    model_name = './pn++/vae_cc_last.pt'
    if is_test:
        model = DeformNet(cycle_consistency=False,pointnet=pointnet).cuda()
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        dataset = get_dataset(args.data_path,['cube','cyl','prism'],)
        test(model,dataset)
    else:
        cycle_consistency = True
        model = DeformNet(cycle_consistency=cycle_consistency,pointnet=pointnet).cuda()
        train(model,writer,cycle_consistency)