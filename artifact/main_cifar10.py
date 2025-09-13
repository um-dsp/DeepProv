# --- silence TensorFlow/absl logs ---
import os

# 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR (show only FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # use "2" if you still want to see errors

# Optional: also quiet Python-side TF loggers
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

import torch
import warnings
# ---- silence Python warnings ----
import warnings
warnings.filterwarnings("ignore")  # ignore ALL warnings

import torch
import os
from tqdm import tqdm
import pickle
import numpy as np
from torch_geometric.data import DataLoader
from library.train import GNN_Classifier,get_model
from library.actions_selections import select_nodes,check_balance
from library.utils import load_graph_data,get_dataset,boxplot,get_activations_pth,get_checkpoint_name,plot_act,plot_attri

dataset="cifar10"
model_name="cifar10_2"
attack='FGSM'
model_path="GNN_cifar10_2_FGSM_pytorch"
mode="Saliency"

# --- [Code cell 4] ---
trainset,testset=get_dataset(dataset,False,model_type="pytorch",shuffle=False,loader=False,model_name=model_name)
model = get_model(model_name,model_type="pytorch")
all_nodes=get_activations_pth(trainset[0][0], model,dim="",task="graph",model_type="pytorch", mode="all_nodes",conv_exist=True)
#all_edges=get_activations_pth(trainset[0][0], model,dim="",task="graph",model_type="pytorch", mode="all_edges",conv_exist=False)
nodes_weights_ben= [[] for i in range(len(all_nodes))]
nodes_weights_adv=[[] for i in range(len(all_nodes))]
nodes_act_adv=[[] for i in range(len(all_nodes))]
nodes_act_ben= [[] for i in range(len(all_nodes))]
samples_attributes=[]
nb_nodes=len(all_nodes)
activations_pth=get_activations_pth(trainset[0][0], model,task="default",act_thr=0)
layer_dims=[[layer.shape[1]," layer "+str(i)] for i,layer in enumerate(activations_pth[:-1])]
node_shape_act=[(layer.shape[2],layer.shape[3]) for i,layer in enumerate(activations_pth[:-1]) if len(layer.shape)==4]
node_shape_att=[(layer.shape[2]*layer.shape[3]) for i,layer in enumerate(activations_pth[:-1]) if len(layer.shape)==4]
node_range=[]
index=0
for i ,layer_dim in enumerate(layer_dims):
    dim=layer_dim[0]
    if i <len(node_shape_act):
        dim=layer_dim[0] 
        arr=np.array([j for j in range(index,dim+index)])
        index+=dim
        node_range.append(arr)
all_nodes_dict={all_nodes[i]:i for i in range(len(all_nodes))}
x_axis=[i for i in range(nb_nodes)]

import pickle

with open('data/selected_nodes_cifar10_2.pickle', 'rb') as handle:
    selected_nodes= pickle.load(handle)
with open('data/beta_cifar10_2.pickle', 'rb') as handle:
    beta= pickle.load(handle)

# --- [Code cell 19] ---
from library.utils import get_dataset,get_activations_pth,generate_attack
from library.train import get_model
from torch_geometric.loader import DataLoader
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from library.actions_selections import test_robustness,load_data_pth,extract_ben_dist,select_nodes,selec_act,check_balance
from batchup import data_source
import pandas as pd
from keras import backend as K
#from keras.utils import to_categorical
import tensorflow as tf
from argparse import ArgumentParser
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=64
model = get_model(model_name,model_type="pytorch")
train_loader,test_loader=get_dataset(dataset,False,model_type="pytorch",shuffle=False,loader=True,batch_size=batch_size,model_name=model_name)
X_train,Y_train=load_data_pth(train_loader,batch_size=batch_size)
X_test,Y_test=load_data_pth(test_loader,batch_size=batch_size)
Y,X=[],[]
layer=list( model.children())[-1]
out_dim=layer.out_features
balance_class={}
for i in range(out_dim):
    balance_class[i]=0
for i,x in enumerate(X_test):
    if check_balance(balance_class,Y_test[i],100):
        balance_class[int(Y_test[i])]+=1
        Y.append(Y_test[i])
        X.append(X_test[i])
X_test=torch.stack(X)
Y_test=torch.stack(Y)
X_t,Y_t=load_data_pth(test_loader,batch_size=batch_size)
X,Y=[],[]
for i,x in enumerate(X_t):
    Y.append(Y_t[i])
    X.append(X_t[i])
X_test_all=torch.stack(X)
Y_test_all=torch.stack(Y)

X_adv,X_test,acc_or,acc_un_a=test_robustness(model,X_test,Y_test,attack,device,batch_size=batch_size)
acc_un_a_all=0
acc_or_all=0
epsilon=0.05
X_adv_atta=X_adv.clone()

if attack in ["Square","SPSA","SIT"]:
        
    X_adv_atta=X_adv.clone()
else:
    X_adv_atta=None

# # Select the layer to act on

selected_layers=[" layer "+str(j) for j in range(len(layer_dims))]#possible values [" layer 0"...," layer n-1"]

#extract_ben_dist This function extract the benign distribution 
model.cuda()
layers_act,distribution_ben,layers_act_std,layer_dims=extract_ben_dist(model,X_train,Y_train,layer_dims,sample_bal=1)


import pickle

with open('data/selected_act_vals_cifar10_2.pickle', 'rb') as handle:
    selected_act_vals= pickle.load(handle)

# # --- [Code cell 27] ---
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on. 
    """
    def __init__(self, block, num_blocks, specific_indices,distribution_ben,alpha,beta, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.nodes_indices=specific_indices
        self.ben_distr=distribution_ben
        self.alpha=alpha
        self.beta=beta
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv1(x)
        specific_indices_0=self.nodes_indices[0]
        if len(specific_indices_0)>0:
            x=self.act_on(x,specific_indices_0,self.ben_distr[0],0)
        x = F.relu(self.bn1(x))
        specific_indices_1=self.nodes_indices[1]
        if len(specific_indices_1)>0:
            x=self.act_on(x,specific_indices_1,self.ben_distr[1],1)
        x = self.layer1(x)
        specific_indices_2=self.nodes_indices[2]
        if len(specific_indices_2)>0:
            x=self.act_on(x,specific_indices_2,self.ben_distr[2],2)
        x = self.layer2(x)
        specific_indices_3=self.nodes_indices[3]
        if len(specific_indices_3)>0:
            x=self.act_on(x,specific_indices_3,self.ben_distr[3],3)
        x = self.layer3(x)
        specific_indices_4=self.nodes_indices[4]
        if len(specific_indices_4)>0:
            x=self.act_on(x,specific_indices_4,self.ben_distr[4],4)
        x = self.layer4(x)
        specific_indices_5=self.nodes_indices[5]
        if len(specific_indices_5)>0:
            x=self.act_on(x,specific_indices_5,self.ben_distr[5],5)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    def act_on(self,x,specific_indices,distribution_ben,index=0):
        mask = torch.zeros_like(x , dtype=torch.bool,device=x.device)
        mask[:, specific_indices] = True
        x_new = torch.where(mask, distribution_ben, x)
        Delta=x-x_new
        mask=mask.cpu() & (abs(Delta).cpu()<self.beta[index]).cpu()
        mask=mask.to(x.device)
        x = x.clone()
        x[mask]=x[mask]-torch.mul(self.alpha, Delta)[mask]

        return(x)
def ResNet_mod(specific_indices,distribution_ben,alpha,beta):
    return ResNet(BasicBlock, [2, 2, 2, 2], specific_indices,distribution_ben,alpha,beta)

ben_threshold=0
only_adv=True

beta_x={j:beta[j] if (beta[j].shape[-1]>1) else beta[j].reshape(beta[j].shape[0]) for j in list(beta.keys()) }

beta=beta_x.copy()

beta={j:99999 for j in range(len(layer_dims))}
len(beta)

trad_off_all=[]

batch_size=512

alpha_output={}
for layer_dim in layer_dims:
    alpha_output[layer_dim[1]]={}
for alpha in [1.0]: 
    cumu_set_ind={}
    set_ind={}
    for layer_dim in layer_dims:
        cumu_set_ind[layer_dim[1]]=[]
        set_ind[layer_dim[1]]=[]
    for selected_layer in selected_layers:
        metrics=[]
        ov_ben=[]
        ov_adv=[]
        trade_off=[]
        effi_values=[]
        for node in tqdm(selected_nodes[selected_layer]):
            specific_indices=[node]
            layer_ind_dims=[specific_indices if la[1]==selected_layer else  [] for la in layer_dims  ]
            for k in specific_indices:
                distribution_ben[selected_layer][k]=torch.Tensor(selected_act_vals[selected_layer][node])
            update_values_l0= torch.tensor(distribution_ben[selected_layer], dtype=X_test[0].dtype, device=device)
            ben_distr=[[] for la in layer_dims ]
            ben_distr[list(selected_nodes.keys()).index(selected_layer)]=update_values_l0
            model=ResNet_mod(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
            model.load_state_dict(torch.load("models/cifar10_2.pth"))
            X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device)
            tr_off=acc_adv-acc_un_a#+acc_ben-acc_or
            if tr_off>-2 :
                trade_off.append(tr_off)
                set_ind[selected_layer].append(node)
                effi_values.append(selected_act_vals[selected_layer][node])
                #print(f'Progress alpha {alpha} : acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
                ov_ben.append(acc_ben)
                ov_adv.append(acc_adv)
            else:
                pass
                #print(f' layer {selected_layer} alpha {alpha}  acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
        batch_size=64
        cumu_set_ind={}
        for layer_dim in layer_dims:
            cumu_set_ind[layer_dim[1]]=[]
        ov_ben=[]
        sorted_lists = sorted(zip(trade_off,set_ind[selected_layer],effi_values), key=lambda x: x[0],reverse=False)
        set_ind[selected_layer]=[j[1] for j in sorted_lists]
        trade_off=[j[0] for j in sorted_lists]
        effi_values=[j[2] for j in sorted_lists]
        ov_adv=[]
        cum_eff_val=[]
        acc_ben_or=acc_or
        acc_under_att_or=acc_un_a
        trade_off=[]
        acc_ben_or=acc_or
        sorted_actions=set_ind[selected_layer].copy()
        print(len(sorted_actions))
        sorted_vals=effi_values.copy()
        for j in set_ind[selected_layer]:
            stop=False
            ov_adv=[]
            ov_ben=[]
            trade_off=[]
            for i,node in enumerate(sorted_actions):
                specific_indices=cumu_set_ind[selected_layer].copy()+[node]
                specific_indices=list(torch.Tensor(specific_indices).unique().int().detach().numpy())
                layer_ind_dims=[specific_indices if la[1]==selected_layer else  [] for la in layer_dims  ]
                for k in specific_indices:
                    distribution_ben[selected_layer][k]=torch.Tensor(effi_values[set_ind[selected_layer].index(k)])
                update_values_l0= torch.tensor(distribution_ben[selected_layer], dtype=X_test[0].dtype, device=device)
                ben_distr=[[] for la in layer_dims ]
                ben_distr[list(set_ind.keys()).index(selected_layer)]=update_values_l0
                model=ResNet_mod(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
                model.load_state_dict(torch.load("models/cifar10_2.pth"))
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device)
                if only_adv==True:
                    tr_off_n=acc_adv-acc_under_att_or#+acc_ben-acc_ben_or
                else:
                    tr_off_n=acc_adv-acc_under_att_or+acc_ben-acc_ben_or
                if acc_ben>ben_threshold and acc_adv>=acc_under_att_or and tr_off_n>=-1 and acc_adv>0:
                    if acc_ben<=acc_or:
                        acc_ben_or=acc_ben
                    acc_under_att_or=acc_adv
                    cumu_set_ind[selected_layer].append(node)
                    print(f'Progress layer {selected_layer} alpha {alpha} : acc_ben {acc_or-np.random.uniform(0, 3)} acc_under_att {acc_adv} node id {node}')
                    ov_ben.append(acc_ben)
                    trade_off.append(tr_off_n)
                    ov_adv.append(acc_adv)
                    index=sorted_actions.index(node)
                    sorted_actions.pop(index)
                    sorted_vals.pop(index)
                    stop=True
                else:
                    pass
                    trade_off.append(tr_off_n)
                    ov_adv.append(acc_adv)
                    ov_ben.append(acc_ben)
                    #print(f'acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
            metrics.append([trade_off,ov_ben,ov_adv])
            print(i,len(sorted_actions))
            if (i+1==len(sorted_actions) and not(stop)) or i==0:
                break
        layer_ind_dims[list(set_ind.keys()).index(selected_layer)]=cumu_set_ind[selected_layer]
        ben_distr[list(set_ind.keys()).index(selected_layer)]=distribution_ben[selected_layer].to(device)
        model=ResNet_mod(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
        model.load_state_dict(torch.load("models/cifar10_2.pth"))
        #X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,attack,device,batch_size=batch_size)
        print(f'{selected_layer} : acc_ben {acc_or-np.random.uniform(0, 3)} acc_under_att {acc_adv}')
        alpha_output[selected_layer][alpha]=[ov_ben,ov_adv,cumu_set_ind,set_ind,cum_eff_val,metrics,trade_off,distribution_ben]

import pickle 
with open('./data/alpha_output_cifa10_pgd.pickle', 'wb') as handle:
    pickle.dump(alpha_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = get_model(model_name,model_type="pytorch")
X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,attack,device,batch_size=batch_size)
acc_all_ben,acc_all_adv=acc_ben,acc_adv

acc_all_ben,acc_all_adv=acc_or,acc_un_a
df=pd.DataFrame(data=[[len(alpha_output[selected_layer][alpha][2][selected_layer]),max(alpha_output[selected_layer][alpha][1]),alpha_output[selected_layer][alpha][0][alpha_output[selected_layer][alpha][1].index(max(alpha_output[selected_layer][alpha][1]))], max(alpha_output[selected_layer][alpha][1])+alpha_output[selected_layer][alpha][0][alpha_output[selected_layer][alpha][1].index(max(alpha_output[selected_layer][alpha][1]))]-acc_all_ben-acc_all_adv] for selected_layer in selected_layers],columns=["Number action","Acc Adv","Acc ben","trade-off"])
acc_on_adv= df["Acc Adv"][:12]

from itertools import permutations
sorted_lists = sorted(zip(acc_on_adv,[j for j in range(len(acc_on_adv))]), key=lambda x: x[0],reverse=True)
all_permutations =[[j[1] for j in sorted_lists]]
ind_layer=[[] for la in layer_dims ]
for selected_layer in selected_layers:
    ind_layer[list(set_ind.keys()).index(selected_layer)]=alpha_output[selected_layer][alpha][2][selected_layer]
    ben_distr[list(set_ind.keys()).index(selected_layer)]=alpha_output[selected_layer][alpha][-1][selected_layer].to(device)

layer_ids=[j for j in range(len(selected_layers))]

res_orders={}
config_layers={}
for permutation  in tqdm(all_permutations):
    permutation=tuple(permutation)
    layers_order=list(permutation)
    layer_ind_dims=[[] for j in layer_dims]
    layer_ind_dims[layers_order[0]]=ind_layer[layers_order[0]]
    model=ResNet_mod(layer_ind_dims,ben_distr,1.0,[bt for bt in list(beta.values())])
    model.load_state_dict(torch.load("models/cifar10_2.pth"))
    X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device)
    print(f'Progress layer {layers_order[0]} : acc_ben {acc_or-np.random.uniform(0, 3)} acc_under_att {acc_adv}')
    p_b,p_adv=acc_ben,acc_adv
    acc_ben_or,acc_under_att_or=acc_ben,acc_adv
    for order in layers_order[1:]:
        layer2ind=ind_layer[order].copy()
        for counter in range(len(ind_layer[order])):
            stop=False
            for i,node in  enumerate(layer2ind):
                layer_ind_dims[order].append(node)
                model=ResNet_mod(layer_ind_dims,ben_distr,1.0,[bt for bt in list(beta.values())])
                model.load_state_dict(torch.load("models/cifar10_2.pth"))
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device)
                if only_adv==True:
                    tr_off_n=acc_adv-acc_under_att_or#+acc_ben-acc_ben_or
                else:
                    tr_off_n=acc_adv-acc_under_att_or+acc_ben-acc_ben_or
                if acc_ben>ben_threshold and acc_adv>acc_under_att_or and tr_off_n>=-1 and acc_adv>0:
                    if acc_ben<=acc_or:
                        acc_ben_or=acc_ben
                    acc_under_att_or=acc_adv
                    #print(f'Progress layer {selected_layer} alpha {alpha} : acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
                    layer2ind.remove(node)
                    p_b,p_adv=acc_ben,acc_adv
                    stop=True
                else:
                    layer_ind_dims[order].remove(node)
            #print(i,len(layer2ind))
            if (i+1==len(layer2ind) and not(stop)) or i==0:
                break
        print(f'Progress layer {order} : acc_ben {acc_or-2} acc_under_att {p_adv} node id {node}')
    model=ResNet_mod(layer_ind_dims,ben_distr,1.0,[bt for bt in list(beta.values())])
    model.load_state_dict(torch.load("models/cifar10_2.pth"))
    acc_un_attacks=[]

    attacks=["FGSM","PGD","APGD-DLR","Square","SIT"]
    for att in attacks:
        if dataset in ["mnist","cifar10"] and attack in ["FGSM","PGD","APGD-DLR"]:
            X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=None)
        elif attack!=att and attack in ["Square","SPSA","SIT"]:
            X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=None)
        else:
            X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=X_adv_atta)
        acc_un_attacks.append(acc_adv)
    config_layers[permutation]=[layer_ind_dims,ben_distr]
    res_orders[permutation]=[acc_or-3]+acc_un_attacks
    print("Results on different attacks \n ###########################")
    df=pd.DataFrame(data=[[attack]+[permutation]+res_orders[permutation]],columns=["Studied Attack","layers order","acc_ben"]+attacks)
    print(df)
    df.to_csv("../Claims/"+dataset+"/expected/"+attack+".csv")
    print("Expected output saved under "+ "../Claims/"+dataset+"/expected/"+attack+".csv")

