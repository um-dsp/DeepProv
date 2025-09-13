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

import os
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from library.train import get_model
from library.utils import load_graph_data,get_dataset,boxplot,get_activations_pth,get_checkpoint_name,plot_act,plot_attri,load_attri
from library.actions_selections import test_robustness,load_data_pth,extract_ben_dist,select_nodes,selec_act,check_balance,evaluate_model
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from library.mod_models import NeuralMnist_v1,Ember_model,MLPClassifierPyTorch_V1

def  load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta):
    if "mnist" in model_name:
        model=NeuralMnist_v1(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
    elif "ember" in model_name:
        model=Ember_model(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
    elif "cuckoo" in model_name:
        model=MLPClassifierPyTorch_V1(layer_ind_dims,ben_distr,alpha,[bt for bt in list(beta.values())])
    model.load_state_dict(torch.load("models/"+model_name+".pth"))
    return(model)




def attri_threshold(attack,dataset):
    if dataset=="mnist":
        if "APGD-DLR"==attack :
            attr_thr_adv=[2.8e-4,1.75e-5,3e-6]
            attr_thr_ben=[2.8e-4,1.75e-5,3e-6]
        elif "FGSM"==attack :
            attr_thr_adv=[5e-4,6e-5,0.8e-5]
            attr_thr_ben=[5e-4,6e-5,0.8e-5]
        elif "PGD"==attack :
            attr_thr_adv=[2.5e-4,1e-5,1.5e-5]
            attr_thr_ben=[1.5e-3,1e-5,1.5e-5]
        elif "Square"==attack :
            attr_thr_adv=[1e-3,6e-6,1e-5]
            attr_thr_ben=[2.5e-4,6e-6,1e-5]
    elif dataset=="ember":
        attr_thr_adv=[0.003,1.4e-3,6e-4,6e-4,8e-4,7.5e-4,5e-4,2e-3,1e-3,1e-4]
        attr_thr_ben=[0.003,1.4e-3,6e-4,6e-4,8e-4,7.5e-4,5e-4,2e-3,1e-3,1e-4]
    elif dataset=="cuckoo":
        attr_thr_adv=[1.5e-2,1.5e-3,1e-3]
        attr_thr_ben=[1e-2,1e-3,1e-3]
    return(attr_thr_ben,attr_thr_adv)
supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks = ['FGSM','CW','PGD',"Bit-Flip","BIM",'Square','APGD-CE','APGD-DLR','Emb-Att',"SPSA","HSJA",None,"SIT"]
pre_trained_models = ['cifar10_1','cuckoo_1','ember_1','mnist_1','mnist_2','mnist_3']
folders = ['Ground_Truth_tf' , 'Benign_tf' ,'Adversarial_tf','Ground_Truth_pth' , 'Benign_pth' ,'Adversarial_pth']
tasks=["graph","default","GNN_explainer"]
model_types=["keras","pytorch"]
expl_modes=["Saliency","IntegratedGradients"]

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', default="mnist", type=str,help=f'supported_dataset= {supported_dataset}')
    parser.add_argument('-model_name', dest='model_name', default="mnist_1", type=str,help=f'pre_trained_models= {pre_trained_models}')
    parser.add_argument('-folder', dest='folder', default="Benign_pth", type=str,help=f'folders= {folders}')
    parser.add_argument('-attack', dest='attack', default=None,help=f'supported_attacks= {supported_attacks}')
    parser.add_argument('-expla_mode', dest='expla_mode', default="Saliency",type=str,help=f'Give the explanation algo{expl_modes}')
    parser.add_argument('-ben_thresh', dest='ben_thresh', default=0,type=int,help=f'Give the Benign threshold ')
    parser.add_argument('-attr_folder', dest='attr_folder', default="attributions_data/",type=str,help=f'Give root folder to save the attributions')
    args = parser.parse_args()

    dataset = args.dataset
    if( dataset not in supported_dataset):
        raise ValueError(f'DeepProv only Supports {supported_dataset}')
        
    model_name = args.model_name
    if(model_name not in pre_trained_models ):
        raise ValueError(f'DeepProv only Supports {pre_trained_models}')
        

    folder = args.folder
    if(folder not in folder):
        raise ValueError(f"DeepProv save folder options are {folder}")

    # Optional Attack argument, if None no attack will be performed on the input

    attack = args.attack
    if(attack not in supported_attacks):
        raise ValueError(f'DeepProv only Supports {supported_attacks}')

    if(not attack and folder =="adversarial"):
      raise ValueError('cannot save adversarial checkpoint without Attack Input')
    return dataset,attack,model_name,folder,args.expla_mode,args.ben_thresh,args.attr_folder






if __name__ == "__main__":
    dataset,attack,model_name,folder,mode,ben_threshold,attri_folder= parseArgs()
 
    print("Loading Attributions and Emperical metrics to select Nodes \n ###########################")
    if dataset=="mnist":
        attacks=["FGSM","PGD","APGD-DLR","square","SIT"]
    else:
        attacks=[attack]
    model_path="GNN_"+model_name+"_"+attack+"_pytorch"
    trainset,testset=get_dataset(dataset,False,model_type="pytorch",shuffle=False,loader=False)
    model = get_model(model_name,model_type="pytorch")
    all_nodes=get_activations_pth(trainset[0][0], model,dim="",task="graph",model_type="pytorch", mode="all_nodes",conv_exist=False)
    all_edges=get_activations_pth(trainset[0][0], model,dim="",task="graph",model_type="pytorch", mode="all_edges",conv_exist=False)
    nb_nodes=len(all_nodes)
    activations_pth=get_activations_pth(trainset[0][0], model,task="default",act_thr=0)
    layer_dims=[[layer.shape[1]," layer "+str(i)] for i,layer in enumerate(activations_pth[:-1])]
    x_axis=[i for i in range(nb_nodes)]

    nodes_weights_ben,nodes_act_ben=load_attri(dataset,model_name,folder,mode,None,all_nodes,model_path,nb_nodes,attri_folder)
    nodes_weights_adv,nodes_act_adv=load_attri(dataset,model_name,folder,mode,attack,all_nodes,model_path,nb_nodes,attri_folder)
    layers_nodes_freq={}
    index=0
    for layer_dim in layer_dims:
        dim=layer_dim[0]
        values_set1 = [[torch.Tensor(i)[torch.Tensor(i)!=0].shape[0]]  for i in nodes_act_ben[index:dim+index]]
        values_set2 = [[torch.Tensor(i)[torch.Tensor(i)!=0].shape[0]]  for i in nodes_act_adv[index:dim+index]]
        array1 = np.array(values_set1)
        array2 = np.array(values_set2)    
        x=(array1 - array2)
        l_inf_norm = np.linalg.norm(x,axis=1, ord=np.inf)
        layers_nodes_freq[layer_dim[1]]=[l_inf_norm]
        index+=dim

    layers_nodes_act={}
    index=0
    beta={}
    for i,layer_dim in enumerate(layer_dims):
        dim=layer_dim[0]
        values_set1 = nodes_act_ben[index:dim+index]
        values_set2 = nodes_act_adv[index:dim+index]
        array1 = np.array(values_set1)
        array2 = np.array(values_set2)
        x=(array1 - array2)
        if len(x.shape)==2:
            beta_vr=x
            original_tensor = torch.tensor(beta_vr)
            # Mask to filter out zeros
            non_zero_mask = abs(original_tensor)>0
            beta_vr = torch.mean(abs(original_tensor * non_zero_mask), dim=1)
        eps=0
        if i==1:
            eps=10
        beta[layer_dim[1]]=beta_vr+beta_vr.mean()+eps
        l_inf_norm = np.linalg.norm(x,axis=1, ord=np.inf)
        layers_nodes_act[layer_dim[1]]=[l_inf_norm]
        #plot_act(l_inf_norm,"l_inf_norm",layer_dims.index(layer_dim))
        index+=dim
    


    #layers_nodes_attri=None
    layers_nodes_attri={}
    index=0
    attri_vals_ben=[]
    attri_vals_adv=[]
    for layer_dim in layer_dims:
        dim=layer_dim[0]
        values_set1 = [float(torch.Tensor(i)[torch.Tensor(i)!=0].mean().detach().numpy()) if len(torch.Tensor(i)[torch.Tensor(i)!=0])>0 else 0.0  for i in nodes_weights_ben[index:dim+index]]
        values_set2 = [float(torch.Tensor(i)[torch.Tensor(i)!=0].mean().detach().numpy()) if len(torch.Tensor(i)[torch.Tensor(i)!=0])>0 else 0.0  for i in nodes_weights_adv[index:dim+index]]
        attri_vals_ben.append(values_set1)
        attri_vals_adv.append(values_set2)
        #plot_attri(values_set1,values_set2,layer_dims.index(layer_dim))
        layers_nodes_attri[layer_dim[1]]=[values_set1,values_set2]
        index+=dim
    attr_thr_ben,attr_thr_adv=attri_threshold(attack,dataset)
    nodes_high_prio,node_t_alter,node_t_null,adv_stat,ben_sat,inter_sta=select_nodes(layer_dims,layers_nodes_freq,layers_nodes_attri,layers_nodes_act,attr_thr_ben,attr_thr_adv,nbr_sam=10000)
    selected_nodes={}
    for layer_dim in layer_dims:
        selected_nodes[layer_dim[1]]=node_t_null[layer_dim[1]]+nodes_high_prio[layer_dim[1]]+node_t_alter[layer_dim[1]]
        layers_nodes_act={}
    attacks=[attack]
    model_paths=[model_path]
    ben_vals_freqs={}
    adv_vals_freqs={}
    ben_vals_acts={}
    adv_vals_acts={}
    ben_vals_attrs={}
    adv_vals_attrs={}
    for jk,attack in enumerate(attacks):
        ben_vals_freq=[]
        adv_vals_freq=[]

        adv_vals_act=[]
        ben_vals_act=[]
        adv_vals_attr=[]
        ben_vals_attr=[]
        index=0
        for i,layer_dim in enumerate(layer_dims):
            dim=layer_dim[0]
            values_set1 = [torch.Tensor(i)[torch.Tensor(i)!=0].shape[0]  for i in nodes_act_ben[index:dim+index]]
            values_set2 = [torch.Tensor(i)[torch.Tensor(i)!=0].shape[0]  for i in nodes_act_adv[index:dim+index]]
            values_set1 = np.array(values_set1)
            values_set2 = np.array(values_set2)
            ben_vals_freq.append(values_set1)
            adv_vals_freq.append(values_set2)
            values_set1 = [float(torch.Tensor(i)[torch.Tensor(i)!=0].mean().detach().numpy()) if len(torch.Tensor(i)[torch.Tensor(i)!=0])>0 else 0.0  for i in nodes_weights_ben[index:dim+index]]
            values_set2 = [float(torch.Tensor(i)[torch.Tensor(i)!=0].mean().detach().numpy()) if len(torch.Tensor(i)[torch.Tensor(i)!=0])>0 else 0.0  for i in nodes_weights_adv[index:dim+index]]
            values_set1 = np.array(values_set1)
            values_set2 = np.array(values_set2)
            ben_vals_attr.append(values_set1)
            adv_vals_attr.append(values_set2)
            values_set1 = torch.tensor(nodes_act_ben[index:dim+index])
            values_set2 = torch.tensor(nodes_act_adv[index:dim+index])
            values_set1=values_set1.mean(axis=-1).detach().numpy()
            values_set2=values_set2.mean(axis=-1).detach().numpy()
            ben_vals_act.append(values_set1)
            adv_vals_act.append(values_set2)
            index+=dim
        
        ben_vals_freqs[attack]=ben_vals_freq
        ben_vals_attrs[attack]=ben_vals_attr
        ben_vals_acts[attack]=ben_vals_act
        adv_vals_attrs[attack]=adv_vals_attr
        adv_vals_acts[attack]=adv_vals_act
        adv_vals_freqs[attack]=adv_vals_freq
    data_layers=[ben_vals_freqs,ben_vals_attrs,ben_vals_acts,adv_vals_attrs,adv_vals_acts,adv_vals_freqs]
    with open('./data/data_layers'+dataset+'.pickle', 'wb') as handle:
        pickle.dump(data_layers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_test_sample_bal={'mnist':1000,'ember':500000,'cuckoo':3000}
    X_train_sample_bal={'mnist':1000,'ember':10000,'cuckoo':10000}
    batch_size=1000
    epsilon=0.05
    model = get_model(model_name,model_type="pytorch")
    train_loader,test_loader=get_dataset(dataset,False,model_type="pytorch",shuffle=False,loader=True,batch_size=batch_size,attack=attack)
    X_train,Y_train=load_data_pth(train_loader,batch_size=batch_size)
    X_test,Y_test=load_data_pth(test_loader,batch_size=batch_size)
    if dataset=="ember":
        X_train =X_train.detach().numpy()
        with open('./data/scaler_ember.pickle', 'rb') as handle:
            scaler= pickle.load(handle)
        X_train_scaled = scaler.transform(X_train)
        X_train=torch.tensor(X_train_scaled, dtype=torch.float)
        epsilon=10000

    Y,X=[],[]
    layer=list( model.children())[-1]
    out_dim=len(Y_test.unique())
    balance_class={}
    for i in range(out_dim):
        balance_class[i]=0
    for i,x in enumerate(X_test):
        if check_balance(balance_class,Y_test[i],X_test_sample_bal[dataset]):
            balance_class[int(Y_test[i])]+=1
            Y.append(Y_test[i])
            X.append(X_test[i])
            
    X_test=torch.stack(X)
    Y_test=torch.stack(Y)
    print("Generating the attack \n ###########################")
    selected_layers=[" layer "+str(j) for j in range(len(layer_dims))]#possible values [" layer 0"...," layer n-1"]
    X_adv,X_test,acc_or,acc_un_a=test_robustness(model,X_test,Y_test,attack,device)
    print("ben acc :",acc_or," adv acc:", acc_un_a)
    if dataset in ["mnist","cifar10"] and attack in ["FGSM","PGD","APGD-DLR"]:
        X_attacked=None
    else:
        X_attacked=X_adv
        beta={j:9999999999 for j in range(len(selected_layers))}
    
    print("Computing Benign Distribution on the Selcted Nodes \n ###########################")
    model = get_model(model_name,model_type="pytorch")
    layers_act,distribution_ben,layers_act_std,layer_dims=extract_ben_dist(model,X_train,Y_train,layer_dims,sample_bal=X_train_sample_bal[dataset])
    selected_act_vals=selec_act(selected_nodes,selected_layers,layers_act,layers_act_std,epsilon,bandwidth=0.1)
    X_train,Y_train=load_data_pth(train_loader,batch_size=batch_size)
    X_train=torch.tensor(X_train, dtype=torch.float)
    Y,X=[],[]
    for i,x in enumerate(X_train[:30000]):
        Y.append(Y_train[i])
        X.append(X_train[i])
    X_train=torch.stack(X)
    Y_train=torch.stack(Y)
    # X_adv_train,X_tr,acc_or_tr,acc_un_a_tr=test_robustness(model,X_train[:30000],Y_train[:30000],attack,device)
    # model = get_model(model_name,model_type="pytorch")
    # #extract_ben_dist This function extract the benign distribution 
    # layers_act_adv,distribution_adv,layers_act_std_adv,layer_dims_adv=extract_ben_dist(model,X_adv_train,Y_train,layer_dims,sample_bal=X_train_sample_bal[dataset])
    # with open('./data/layers_act_adv'+dataset+'.pickle', 'wb') as handle:
    #     pickle.dump(layers_act_adv, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('./data/layers_act'+dataset+'.pickle', 'wb') as handle:
    #     pickle.dump(layers_act, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

    with open('./data/data_layers_'+dataset+'.pickle', 'wb') as handle:
        pickle.dump(data_layers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if dataset=="mnist":

        X_t,Y_t=load_data_pth(test_loader,batch_size=batch_size)
        X,Y=[],[]
        for i,x in enumerate(X_t):
            Y.append(Y_t[i])
            X.append(X_t[i])
        X_test_all=torch.stack(X)
        Y_test_all=torch.stack(Y)
    else:
        X_test_all=X_test
        Y_test_all=Y_test    
    
    alpha_output={}
    print("Robustness Enhancement Process \n ###########################")
    for layer_dim in layer_dims:
        alpha_output[layer_dim[1]]={}
    for alpha in [1.0]: 
        cumu_set_ind={}
        set_ind={}
        for layer_dim in layer_dims:
            cumu_set_ind[layer_dim[1]]=[]
            set_ind[layer_dim[1]]=[]
        for selected_layer in selected_layers:
            ov_ben=[]
            ov_adv=[]
            metrics=[]
            trade_off=[]
            effi_values=[]
            for node in tqdm(selected_nodes[selected_layer]):
                selected_act_vals[selected_layer][node]
                specific_indices=[node]
                layer_ind_dims=[specific_indices if la[1]==selected_layer else  [] for la in layer_dims  ]
                for k in specific_indices:
                    distribution_ben[selected_layer][k]=torch.Tensor(selected_act_vals[selected_layer][node])
                update_values_l0= torch.tensor(distribution_ben[selected_layer], dtype=X_test[0].dtype, device=device)
                ben_distr=[[] for la in layer_dims ]
                ben_distr[list(selected_nodes.keys()).index(selected_layer)]=update_values_l0
                ben_distr[list(selected_nodes.keys()).index(selected_layer)]=update_values_l0
                model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device,X_adv=X_attacked)
                tr_off=acc_adv-acc_un_a#+acc_ben-acc_or
                if tr_off>=-1 :
                    trade_off.append(tr_off)
                    set_ind[selected_layer].append(node)
                    effi_values.append(selected_act_vals[selected_layer][node])
                    #print(f'Progress alpha {alpha} : acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
                    ov_ben.append(acc_ben)
                    ov_adv.append(acc_adv)
                else:
                    pass
                    #print(f' layer {selected_layer} alpha {alpha}  acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
            
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
                    ben_distr[list(selected_nodes.keys()).index(selected_layer)]=update_values_l0
                    model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
                    X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device,X_adv=X_attacked)
                    if ben_threshold==0:
                        tr_off_n=acc_adv-acc_under_att_or#+acc_ben-acc_ben_or
                    else:
                        tr_off_n=acc_adv-acc_under_att_or+acc_ben-acc_ben_or
                    if (attack=="PGD" or attack=="APGD-DLR") and dataset=="mnist":
                        condition=True
                    else:
                        condition=acc_ben>ben_threshold and acc_adv>=acc_under_att_or and tr_off_n>=0
                    if condition:
                        if acc_ben<=acc_or:
                            acc_ben_or=acc_ben
                        acc_under_att_or=acc_adv
                        cumu_set_ind[selected_layer].append(node)
                        #print(f'Progress layer {selected_layer}  : acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
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
                #print(i,len(sorted_actions))
                if (i+1==len(sorted_actions) and not(stop)) or i==0:
                    break
            layer_ind_dims[list(set_ind.keys()).index(selected_layer)]=cumu_set_ind[selected_layer]
            ben_distr[list(set_ind.keys()).index(selected_layer)]=distribution_ben[selected_layer].to(device)
            model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
            X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,attack,device,X_adv=X_attacked)
            print(f'{selected_layer} : acc_ben {acc_ben} acc_under_att {acc_adv}')
            alpha_output[selected_layer][alpha]=[ov_ben,ov_adv,cumu_set_ind,set_ind,cum_eff_val,metrics,trade_off,distribution_ben]
    with open('./data/alpha_output_'+dataset+'.pickle', 'wb') as handle:
        pickle.dump(alpha_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model = get_model(model_name,model_type="pytorch")
    X_advs,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,attack,device)
    acc_all_ben,acc_all_adv=acc_ben,acc_adv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Combining the best actions sequence between different layers \n ###########################")
    acc_on_adv= [alpha_output[selected_layer][alpha][1] for selected_layer in selected_layers]
    sorted_lists = sorted(zip(acc_on_adv,[j for j in range(len(acc_on_adv))]), key=lambda x: x[0],reverse=True)
    ind_layer=[[] for la in layer_dims ]
    for selected_layer in selected_layers:
        ind_layer[list(set_ind.keys()).index(selected_layer)]=alpha_output[selected_layer][alpha][2][selected_layer]
        ben_distr[list(set_ind.keys()).index(selected_layer)]=alpha_output[selected_layer][alpha][-1][selected_layer].to(device)

    layer_ids=[j for j in range(len(layer_dims))]
    all_permutations =[[j[1] for j in sorted_lists]]
    #all_permutations=permutations(layer_ids)
    res_orders={}
    config_layers={}
    for permutation  in tqdm(all_permutations):
        permutation=tuple(permutation)
        layers_order=list(permutation)
        layer_ind_dims=[[] for j in layer_dims]
        layer_ind_dims[layers_order[0]]=ind_layer[layers_order[0]]
        model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
        X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device,X_adv=X_attacked)
        acc_ben_or,acc_under_att_or=acc_ben,acc_adv
        for order in layers_order[1:]:
            layer2ind=ind_layer[order].copy()
            for counter in range(len(ind_layer[order])):
                stop=False
                for i,node in  enumerate(layer2ind):
                    layer_ind_dims[order].append(node)
                    model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
                    X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test,Y_test,attack,device,X_adv=X_attacked)
                    if ben_threshold==0:
                        tr_off_n=acc_adv-acc_under_att_or#+acc_ben-acc_ben_or
                    else:
                        tr_off_n=acc_adv-acc_under_att_or+acc_ben-acc_ben_or
                    if acc_ben>ben_threshold and acc_adv>=acc_under_att_or and tr_off_n>=0 :
                        if acc_ben<=acc_or:
                            acc_ben_or=acc_ben
                        acc_under_att_or=acc_adv
                        #print(f'Progress layer {selected_layer} alpha {alpha} : acc_ben {acc_ben} acc_under_att {acc_adv} node id {node}')
                        layer2ind.remove(node)
                        stop=True
                    else:
                        layer_ind_dims[order].remove(node)
                #print(i,len(layer2ind))
                if i+1==len(layer2ind) and not(stop):
                    break
        X_t,Y_t=load_data_pth(test_loader,batch_size=batch_size)
        X,Y=[],[]
        for i,x in enumerate(X_t):
            Y.append(Y_t[i])
            X.append(X_t[i])
        X_test_all=torch.stack(X)
        Y_test_all=torch.stack(Y)
        if dataset=="mnist":
            attacks=["FGSM","PGD","APGD-DLR","Square","SIT"]
        else:
            attacks=[attack]
        model=load_DP_model(model_name,layer_ind_dims,ben_distr,alpha,beta)
        acc_un_attacks=[]
        for att in attacks:
            if dataset in ["mnist","cifar10"] and attack in ["FGSM","PGD","APGD-DLR"]:
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=None)
            elif attack!=att and attack in ["Square","SPSA","SIT"]:
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=None)
            else:
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=X_attacked)
            acc_un_attacks.append(acc_adv)
        config_layers[permutation]=[layer_ind_dims,ben_distr]
        res_orders[permutation]=[acc_ben]+acc_un_attacks
        print("Results on different attacks \n ###########################")
        df=pd.DataFrame(data=[[attack]+[permutation]+res_orders[permutation]],columns=["Studied Attack","layers order","acc_ben"]+attacks)
        print(df)
        df.to_csv("../Claims/"+dataset+"/expected/"+attack+".csv")
        print("Expected output saved under "+ "../Claims/"+dataset+"/expected/"+attack+".csv")



    #Assessing Best Alpha
    ben_distr=[[] for j in range(len(layer_dims))]

    acc_on_adv= [alpha_output[selected_layer][1][1] for selected_layer in selected_layers]
    sorted_lists = sorted(zip(acc_on_adv,[j for j in range(len(acc_on_adv))]), key=lambda x: x[0],reverse=True)
    ind_layer=[[] for la in layer_dims ]
    for i, selected_layer in enumerate(selected_layers):
        ind_layer[i]=alpha_output[selected_layer][1][2][selected_layer]
        ben_distr[i]=alpha_output[selected_layer][1][-1][selected_layer].to(device)
        
    layer_ids=[j for j in range(len(layer_dims))]
    all_permutations =[[j[1] for j in sorted_lists]]
    res_orders={}
    alpha_metrics={}
    config_layers={}
    for permutation  in tqdm(all_permutations):
        permutation=layer_ids

        layers_order=list(permutation)
        
        for j in range(len(layers_order)):
            alpha_metrics[j]=[]
            for alp in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                layer_ind_dims=[[] for j in layer_dims] 
                layer_ind_dims[layers_order[j]]=ind_layer[layers_order[j]]  
                model=load_DP_model(model_name,layer_ind_dims,ben_distr,alp,beta)
                X_adv,X,acc_ben,acc_adv=test_robustness(model,X_test_all,Y_test_all,att,device,X_adv=X_attacked)
                alpha_metrics[j].append(acc_ben-acc_or+acc_adv-acc_un_a)
    with open('./data/metrics_'+dataset+'.pickle', 'wb') as handle:
        pickle.dump(alpha_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
