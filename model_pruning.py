from model import MESH_NET
import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp


def basic_pruning(mesh_net):
    
    print(mesh_net)
    module = mesh_net.conv1
    print(list(module.named_parameters()))
    parameter_to_prune = (
        (mesh_net.conv1.lin, 'weight'),
        (mesh_net.conv2.lin, 'weight'), 
        (mesh_net.conv3.lin, 'weight'),
    )
    prune.ln_structured(mesh_net.conv1.lin, 'weight', 0.3, 1, 0)
    prune.remove(mesh_net.conv1.lin, 'weight')
    # prune.remove(mesh_net.conv2.lin, 'weight')
    # prune.remove(mesh_net.conv3.lin, 'weight')
    print(list(module.named_parameters()))
    model_dir = 'Models/MESH2IR'
    torch.save(
        mesh_net.state_dict(),
        '%s/mesh_net_epoch_40_pruned.pth' % (model_dir)
    )
    
def torch_pruning(model):
    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.MagnitudeImportance()

    ignored_layers = []
    for m in model.modules():
        if (isinstance(m, torch.nn.Linear) and m.out_features == 8) or (isinstance(m, torch.nn.BatchNorm1d)) or isinstance(m, torch.nn.ReLU):
            ignored_layers.append(m) # DO NOT prune the final classifier!
    iterative_steps = 5 # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        
    model.zero_grad()
    torch.save(model, 'Models/MESH2IR/mesh_net_epoch_40_tp.pth')

            
if __name__=='__main__':
    mesh_net = MESH_NET() 
    mesh_net_path = 'Models/MESH2IR/mesh_net_epoch_40.pth'

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
    basic_pruning(mesh_net)