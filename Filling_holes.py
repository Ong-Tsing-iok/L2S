import trimesh

mesh = trimesh.load('Meshes/merged_mesh.obj', force='mesh')
print(mesh.is_watertight)
mesh.fill_holes()
mesh.export('Meshes/filled_mesh_by_trimesh.obj')