import torch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from batchup import data_source
import numpy as np
import torch.nn as nn

from torch_geometric.loader import DataLoader
from library.utils import generate_attack,get_activations_pth
def weighted_mean(x_vals, density_values):
    # The mean is the sum of the product of values and their probabilities (density values in this case)
    density_normalized = density_values / np.sum(density_values)
    return np.sum(x_vals * density_normalized)

def weighted_std(x_vals, density_values, mean):
    # Variance is the sum of the squared differences from the mean, weighted by the probabilities
    density_normalized = density_values / np.sum(density_values)
    variance = np.sum((x_vals - mean) ** 2 * density_normalized)
    return np.sqrt(variance)
def apply_kde(indexed_data,std_bool,bandwidth):
    subset_data,index = indexed_data
    if len(index)==1:
        index=index[0][0]
        s_bool=std_bool
    else:
        s_bool=std_bool[index][0]
    if s_bool==False:
        subset_data = subset_data.reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(subset_data)
        x_vals = np.linspace(min(subset_data), max(subset_data), 100)
        log_density = kde.score_samples(x_vals.reshape(-1, 1))
        density_values = np.exp(log_density)
        #mean_ben = weighted_mean(x_vals, density_values)
        #std_ben = weighted_std(x_vals, density_values, mean_ben)
        peaks_regions=density_values[density_values>(density_values.max()/2)]
        peaks_regions_vals=x_vals[density_values>(density_values.max()/2)]
        res=np.random.choice(peaks_regions_vals[:,0], size=1, p=(peaks_regions/sum(peaks_regions)))
    else:
        res=[0]
    return res

def selec_act(selected_nodes,selected_layers,layers_act,layers_act_std,epsilon,bandwidth=0.1):
    selected_act_vals={}
    for selected_layer in selected_layers:
        selected_act_vals[selected_layer]={}
        for node in tqdm(selected_nodes[selected_layer]):
            std_node=np.array(layers_act_std[selected_layer][node])
            std_bool=std_node<epsilon
            data=layers_act[selected_layer][node].detach().numpy()
            axis=len(data.shape)-1
            data=np.stack(data,axis=axis)
            activation_val=np.zeros(std_node.shape)
            activation_val[std_bool]=data.mean(axis=axis)[std_bool]
            kde_results = np.apply_along_axis(lambda x: apply_kde((x,np.where(np.all(data == x, axis=axis))),std_bool,bandwidth), axis=axis, arr=data)
            activation_val[std_bool==False]=kde_results.reshape(std_bool.shape )[std_bool==False]
            selected_act_vals[selected_layer][node]=activation_val
    return(selected_act_vals)
def extract_ben_dist(model,X,Y,layer_dims,sample_bal=6000):
    layers_act={}
    index=0
    for layer_dim in layer_dims:
        layers_act[layer_dim[1]]=[]
    layer=list( model.children())[-1]
    out_dim=len(Y.unique())
    balance_class={}
    for i in range(out_dim):
        balance_class[i]=0
    for i,x in enumerate(X):
        if check_balance(balance_class,Y[i],sample_bal):
            balance_class[int(Y[i])]+=1
            x=x.cuda(0)
            model=model.cuda(0)
            activations_pth=get_activations_pth(x, model,task="default",act_thr=0)
            for j ,layer_dim in  enumerate(layer_dims):
                layers_act[layer_dim[1]].append(activations_pth[int(layer_dim[1].split(" layer ")[1])][0].cpu())
    distribution_ben={}
    for layer_dim in layer_dims:
        distribution_ben[layer_dim[1]]=layers_act[layer_dim[1]][0].clone()
        layers_act[layer_dim[1]]=torch.stack(layers_act[layer_dim[1]], dim=1)
    layers_act_std={}
    for layer_dim in layer_dims:
        layers_act_std[layer_dim[1]]=[]
    for layer_dim in layer_dims:
        for j in range(layers_act[layer_dim[1]].shape[0]):
            data=layers_act[layer_dim[1]][j].detach().numpy()
            axis=len(data.shape)-1
            data=np.stack(data,axis=axis)
            std_dev = np.std(data,axis=axis)
            nn_zr=np.array(std_dev)!=0
            dim=len(nn_zr.shape)
            if dim>0:
                min_value = np.min(data,axis=axis)
                max_value = np.max(data,axis=axis)
                nr_data=nn_zr[:, :, np.newaxis]*(data-min_value[:, :, np.newaxis])
                numerator=nn_zr[:, :, np.newaxis]*(max_value[:, :, np.newaxis]-min_value[:, :, np.newaxis])
                normalized_data_minmax = (nr_data) / (numerator)
                std_dev = np.std(normalized_data_minmax,axis=axis)
            elif std_dev!=0:
                min_value = np.min(data)
                max_value = np.max(data)
                normalized_data_minmax = (data - min_value) / (max_value - min_value)
                std_dev = np.std(normalized_data_minmax)
                normalized_data_minmax = (data - min_value) / (max_value - min_value)
                std_dev = np.std(normalized_data_minmax,axis=axis)
            layers_act_std[layer_dim[1]].append(std_dev)

    return(layers_act,distribution_ben,layers_act_std,layer_dims)

def select_nodes(layer_dims,layers_nodes_freq,layers_nodes_attri,layers_nodes_act,attr_thr_ben,attr_thr_adv,nbr_sam=10000):
    nodes_high_prio={}
    node_t_alter={}
    node_t_null={}
    for layer_dim in layer_dims:
        dim=layer_dim[1]
        nodes_high_prio[dim]=[]
        node_t_alter[dim]=[]
        node_t_null[dim]=[]
    for layer_dim in layer_dims:
        dim=layer_dim[1]
        for node in range(layer_dim[0]):
            if layers_nodes_freq[dim][0][node]==nbr_sam:
                node_t_null[dim].append(node)
            if layers_nodes_act[dim][0][node]==0:
                continue
            if layers_nodes_attri[dim]!=None:
                if layers_nodes_attri[dim][0][node]>=attr_thr_ben[layer_dims.index(layer_dim)]:
                    if layers_nodes_attri[dim][1][node]>=attr_thr_adv[layer_dims.index(layer_dim)]:
                        nodes_high_prio[dim].append(node)
                    else:
                        node_t_alter[dim].append(node)
                else:
                    node_t_alter[dim].append(node)
            else:
                node_t_alter[dim].append(node)
    adv_nodes={}
    ben_nodes={}
    nodes_inter={}
    non_adv={}
    for layer_dim in layer_dims:
        dim=layer_dim[1]
        adv_nodes[dim]=[]
        ben_nodes[dim]=[]
        nodes_inter[dim]=[]
        non_adv[dim]=[]
    for layer_dim in layer_dims:
        dim=layer_dim[1]
        for node in range(layer_dim[0]):
            if layers_nodes_attri[dim]!=None:
                if layers_nodes_attri[dim][0][node]>=attr_thr_ben[layer_dims.index(layer_dim)]:
                    if layers_nodes_attri[dim][1][node]>=attr_thr_adv[layer_dims.index(layer_dim)]:
                        nodes_inter[dim].append(node)
                    else:
                        ben_nodes[dim].append(node)
            if layers_nodes_attri[dim][1][node]<attr_thr_adv[layer_dims.index(layer_dim)]:
                non_adv[dim].append(node)
        adv_nodes[dim]=list(set([node for node in range(layer_dim[0])]).difference(set(ben_nodes[dim]+nodes_inter[dim]+non_adv[dim])))
    adv_stat=sum([len(data) for data in adv_nodes.values()])
    ben_sat=sum([len(data) for data in ben_nodes.values()])
    inter_sat=sum([len(data) for data in nodes_inter.values()])
    return nodes_high_prio,node_t_alter,node_t_null,adv_stat,ben_sat,inter_sat
def load_data_pth(data_loader,batch_size=1000):
    dims=list(data_loader.dataset[0][0].shape)
    shape=(len(data_loader.dataset),)
    for dim in dims :
        shape+=(dim,)
    X=torch.zeros(shape)
    if  str(type(data_loader.dataset[0][1])) =="<class 'torch.Tensor'>"  and len(data_loader.dataset[0][1].shape)>0  and data_loader.dataset[0][1].shape[0]==1:
        Y=torch.zeros((len(data_loader.dataset),1))
    else:    
        Y=torch.zeros(len(data_loader.dataset))
    i=0
    for x,y in data_loader:
        Y[i * batch_size:(i + 1) * batch_size]=y
        X[i * batch_size:(i + 1) * batch_size]=x
        i+=1
    return(X,Y)
def evaluate_model(model, test_loader,device='cpu'):

    '''Evaluate a PyTorch model'''
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    #model = nn.DataParallel(model)

    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data in (test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            if images.shape[-1]==2381:
                predicted=((outputs) > 0.5).float()
                predicted=predicted.view(-1).cpu().numpy()
                labels=labels.view(-1).cpu().numpy()
            elif outputs.data.shape[1]==1:
                predicted=((outputs) > 0.5).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #print(f'Accuracy of the model on the test images: {accuracy}%')
    return(accuracy)
import pickle
def check_balance(balance_class,y,nbr_samples_per_class):
    return(balance_class[int(y)]<nbr_samples_per_class)

def test_robustness(model,X,Y,attack,device,batch_size=1000,X_adv=None):
    model=model.to(device)
    if attack=="EMBER":
        X_ben=X.clone()
        import pickle
        with open('./data/scaleremb.pickle', 'rb') as handle:
            scaler= pickle.load(handle)
        x=torch.tensor(scaler.transform(X_ben.detach().numpy()))
    else:
        x=X.clone()
    dataset=[(x,y) for x,y in zip(x,Y)]
    dataset= DataLoader(dataset, batch_size=batch_size, shuffle=False)
    acc_ben=evaluate_model(model,dataset,device=device)
    dataset=[(x,y) for x,y in zip(X,Y)]
    dataset= DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if(attack):
        if  X_adv==None:
            X_adv=0
            X_adv=X.clone()
            i = 0
            for (batch_X, batch_y) in dataset:
                #print('Generating {} from sample {} to {}'.format(attack, i * batch_size, (i + 1) * batch_size - 1))
                batch_X=batch_X.to(device)
                adv_batch_X = generate_attack(model, batch_X, batch_y, attack,model_type="pytorch")
                adv_batch_X=adv_batch_X.to("cpu")
                #dataset=[(x,y) for x,y in zip(adv_batch_X,batch_y)]
                #dataset= DataLoader(dataset, batch_size=64, shuffle=True)
                #evaluate_model(model,dataset,device=device)
                X_adv[i * batch_size:(i + 1) * batch_size] = adv_batch_X
                i += 1
        dataset=[(x,y) for x,y in zip(X_adv,Y)]
        dataset= DataLoader(dataset, batch_size=batch_size, shuffle=True)
        acc_adv=evaluate_model(model,dataset,device=device)
    return(X_adv,X,acc_ben,acc_adv)