import torch
import ipdb
import pytorch3d
from pytorch3d import loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# loss_chamfer = 
	dists_src_tgt = pytorch3d.ops.knn_points(p1=point_cloud_src,p2=point_cloud_tgt)
	dists_tgt_src = pytorch3d.ops.knn_points(p1=point_cloud_tgt,p2=point_cloud_src)
	loss_chamfer = torch.mean(dists_src_tgt[0]) + torch.mean(dists_tgt_src[0])

	return loss_chamfer

def kl_loss(mean_z,var):

	# ipdb.set_trace()
	loss_kl = torch.mean(0.5*torch.sum(-torch.ones_like(mean_z)-torch.log(1e-8+var**2)+var**2+mean_z**2,dim=1),dim=0)
	return loss_kl


def smoothness_loss(mesh_src):
	loss_laplacian = loss.mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian