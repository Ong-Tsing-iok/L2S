import os
import time
import json
import numpy as np
import random
import argparse
import pickle

import torch
import torch_geometric
from torch_geometric.io import read_ply, read_obj

path = "Simplified_Meshes/"
move_path = "Mesh_Graphs/"

if not os.path.isdir(move_path):
	os.mkdir(move_path)
mesh_files = os.listdir(path)
print("mesh files ",mesh_files)
for file in mesh_files:
	
	if(file.endswith(".obj")):
		start_time = time.time()
		full_mesh_path = path +  file 
		graph_path = move_path +  file[0:len(file)-4] +".pickle"
		print("graph_path ",graph_path)
		if(os.path.exists(full_mesh_path)):
			print("came here ")
			mesh = read_obj(full_mesh_path);
			print(mesh)
			pre_transform = torch_geometric.transforms.FaceToEdge();
			graph =pre_transform(mesh);
			print(graph['edge_index'])

			with open(graph_path, 'wb') as f:
				pickle.dump(graph, f, protocol=2)
		end_time = time.time()
		print(f'graph_generate_time: {end_time - start_time}')
		



