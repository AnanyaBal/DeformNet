import openmesh
import pytorch3d
import numpy as np
import glob


for f in glob.glob('/home/cobra/abhimanyu_course/DeformNet/cube_222_sw.STL'):
	filename = f.split('/')[-1]
	filename = filename.split('.')[0]
	print(filename)
	mesh = openmesh.read_trimesh(f)
	openmesh.write_mesh(mesh=mesh,filename='/home/cobra/abhimanyu_course/DeformNet/{}.obj'.format(filename))