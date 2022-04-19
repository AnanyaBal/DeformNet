import pytorch3d
import torch
from pytorch3d.renderer import (
	AlphaCompositor,
	RasterizationSettings,
	MeshRenderer,
	MeshRasterizer,
	PointsRasterizationSettings,
	PointsRenderer,
	PointsRasterizer,
	HardPhongShader,
)
from pytorch3d.io import load_obj, load_ply
import ipdb
import imageio
import tqdm

def get_device():
	"""
	Checks if GPU is available and returns device accordingly.
	"""
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")
	return device


def load_mesh(path):

	# ipdb.set_trace()
	verts, faces, _ = load_obj(path,load_textures=False)
	# verts, faces,  = load_ply(path)
	faces = faces.verts_idx
	# ipdb.set_trace()
	center = verts.mean(0)
	verts = verts - center
	scale = max(verts.abs().max(0)[0])
	verts = verts / scale
	return verts, faces

def get_mesh(verts,faces):
	ipdb.set_trace()
	return pytorch3d.structures.Meshes(verts, faces)

def get_mesh_renderer(image_size=512, lights=None, device=None):

	if device is None:
		if torch.cuda.is_available():
			device = torch.device("cuda:0")
		else:
			device = torch.device("cpu")
	raster_settings = RasterizationSettings(
		image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
	)
	renderer = MeshRenderer(
		rasterizer=MeshRasterizer(raster_settings=raster_settings),
		shader=HardPhongShader(device=device, lights=lights),
	)
	return renderer


def render_mesh(mesh_path=None, image_size=1024, color=[0.7, 0.7, 1], device=None,azimuth=0):
	if device is None:
		device = get_device()

	renderer = get_mesh_renderer(image_size=image_size)

	vertices, faces = load_mesh(mesh_path)
	vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
	faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
	textures = torch.ones_like(vertices)  # (1, N_v, 3)
	textures = textures * torch.tensor(color)  # (1, N_v, 3)
	mesh = pytorch3d.structures.Meshes(
		verts=vertices,
		faces=faces,
		textures=pytorch3d.renderer.TexturesVertex(textures),
	)
	mesh = mesh.to(device)

	R, T = pytorch3d.renderer.look_at_view_transform(dist=-3, elev=0, azim=azimuth)
	cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
  
	lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

	rend = renderer(mesh, cameras=cameras, lights=lights)
	rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
	# ipdb.set_trace()
	return rend

def gif_maker(im_list):
	my_images = im_list  # List of images [(H, W, 3)]
	imageio.mimsave('mesh_turntable_undeformed.gif', my_images, fps=15)

if __name__ == "__main__":
	im_list=[]
	mesh_path = './data/undeformed/obj_0.obj'
	for az in tqdm.tqdm(range(0, 370, 10)):
		image = render_mesh(mesh_path=mesh_path, image_size=256, azimuth=az)
		im_list.append(image)
	gif_maker(im_list)