from torch.autograd import Variable
from torch_geometric.loader import DataLoader
import numpy as np
import pickle
import os
import torch
from torch import nn
from wavefile import WaveWriter
import uuid



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_network_stageI(netG_path,mesh_net_path):
        from model import STAGE1_G, STAGE1_D, MESH_NET
        netG = STAGE1_G()
        netG.apply(weights_init)

        mesh_net =MESH_NET() 

        if netG_path!= '':
            state_dict = \
                torch.load(netG_path,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', netG_path)
       
        if mesh_net_path != '':
            state_dict = \
                torch.load(mesh_net_path,
                           map_location=lambda storage, loc: storage)
            new_dict = {}
            for k, v in state_dict.items():
                if k == 'pool1.weight' or k == 'pool2.weight' or k == 'pool3.weight':
                    k = f'{k[:6]}select.{k[6:]}'
                new_dict[k] = v
            mesh_net.load_state_dict(new_dict)
            print('Load from: ', mesh_net_path)

        
        netG.cuda()
        mesh_net.cuda()
        return netG, mesh_net
    
def get_graph(full_graph_path):
        
    with open(full_graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    return graph #edge_index, vertex_position
    
def init_models():

    netG_path = "Models/MESH2IR/netG_epoch_40.pth"
    mesh_net_path = "Models/MESH2IR/mesh_net_epoch_40_pruned.pth"
    gpus =[0,1]

    batch_size = 1
    fs = 16000
    netG, mesh_net = load_network_stageI(netG_path,mesh_net_path)
    netG.eval()
    mesh_net.eval()

    netG.to(device='cuda')
    mesh_net.to(device='cuda')

    full_graph_path = "Mesh_Graphs/scene0000_02_new_simplified.pickle"

    data_single = get_graph(full_graph_path)
    data_list=[data_single]*batch_size
    loader = DataLoader(data_list, batch_size=batch_size)

    data = next(iter(loader))
    data['edge_index'] = Variable(data['edge_index'])
    data['pos'] = Variable(data['pos'])
    data = data.cuda()
    mesh_embed = nn.parallel.data_parallel(mesh_net, data,  [gpus[0]])
    return mesh_embed, netG
    
def evaluate(mesh_embed, netG, source, receiver, store = False):
        
    output_directory ="/mnt/c/Users/asdof/Home/Homeworks/multimedia_network/final_project/L2S/Output_one_IR/"

    gpus =[0,1]

    batch_size = 1
    fs = 16000


    if(not os.path.exists(output_directory)):
        os.mkdir(output_directory)

    output_embed  = output_directory+'test_embed'
    if(not os.path.exists(output_embed)):
        os.mkdir(output_embed)

    print("embed_name   ",output_embed)
    
    txt_embedding_list = []
    folder_name_list =[]
    wave_name_list = []

    # source_receiver = [3.1441, 1.3932, 1.6698, 1.224, 5.666, 1.315]
    source_receiver = source + receiver
    txt_embedding_single = np.array(source_receiver).astype('float32')

    txt_embedding_list.append(txt_embedding_single)
    folder_name_list.append('xyz')
    name = str(uuid.uuid4())
    wave_name_list.append(name+".wav")


    txt_embedding =torch.from_numpy(np.array(txt_embedding_list))
    txt_embedding = Variable(txt_embedding)
    txt_embedding = txt_embedding.cuda()

    
    inputs = (txt_embedding,mesh_embed)
    lr_fake, fake, _ = nn.parallel.data_parallel(netG, inputs, [gpus[0]])

    for i in range(len(fake)):
        if(not os.path.exists(output_embed+"/"+folder_name_list[i])):
            os.mkdir(output_embed+"/"+folder_name_list[i])

        fake_RIR_path = output_embed+"/"+folder_name_list[i]+"/"+wave_name_list[i]
        fake_IR = np.array(fake[i].to("cpu").detach())
        
        fake_IR_only = fake_IR[:,0:(4096-128)]
        fake_energy = np.median(fake_IR[:,(4096-128):4096]) * 10*5
        fake_IR = fake_IR_only*fake_energy
        if store:
            f = WaveWriter(fake_RIR_path, channels=2, samplerate=fs)
            f.write(np.array(fake_IR))
            f.close()
        return fake_IR
            

if __name__ == '__main__':
    mesh_embed, netG = init_models()
    evaluate(mesh_embed, netG, [3.1441, 1.3932, 1.6698], [1.224, 5.666, 1.315], store = True)
    evaluate(mesh_embed, netG, [3., 1.32, 1.6698], [1.224, 5.66, 2.315], store = True)
