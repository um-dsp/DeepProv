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

from library.Accessor import Accessor
from library.attributionUtils import adversarial_detection_set,get_attributes
import torch
from library.train import train_on_activations,evaluate_GNN,GNN_Classifier,train_GNN,HGT,get_model,model_output
from library.utils import load_graph_data,get_dataset,get_checkpoint_name,check_conv_layer
import sys

import os
# ---- silence Python warnings ----
import warnings
warnings.filterwarnings("ignore")  # ignore ALL warnings

import pickle
from torch_geometric.loader import DataLoader
import numpy as np
from argparse import ArgumentParser

supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks = ['FGSM','CW','PGD',"Bit-Flip","BIM",'Square','APGD-CE','APGD-DLR','Emb-Att',"SPSA","HSJA","SIT",None]
pre_trained_models = ['cifar10_2','cuckoo_1','ember_1','mnist_1','mnist_2','mnist_3']
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
    parser.add_argument('-task', dest='task', default="default",type=str,help=f'tasks= {tasks}')
    parser.add_argument('-model_type', dest='model_type', default="pytorch",type=str,help=f'model_types= {model_types}')
    parser.add_argument('-model_path', dest='model_path', default="",type=str,help=f'give the model path')
    parser.add_argument('-epochs', dest='epochs', default=20,type=int,help=f'give number of epochs')
    parser.add_argument('-save', dest='save', default=False,type=bool,help=f'Save the trained GNN model [True,False]')
    parser.add_argument('-data_size', dest='data_size', default=1,type=int,help=f'Give the  number of batch to use for train')
    parser.add_argument('-expla_mode', dest='expla_mode', default="Saliency",type=str,help=f'Give the explanation algo{expl_modes}')
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
    task=args.task
    if(task not in tasks ):
        raise ValueError(f'DeepProv only Supports the following tasks {tasks}')
    model_type=args.model_type
    if(model_type not in model_types ):
        raise ValueError(f'DeepProv only Supports the following model types {model_types}')
    model_path=args.model_path
    return dataset,attack,model_name,folder,task,model_type,model_path,args.epochs,args.save,args.data_size,args.expla_mode,args.attr_folder
#We evaluate the GNN model over adersarial and benign test samples
def GNN_eval(model,dataset,model_name,attack,model_type):
    over_acc=0
    if model_type=="pytorch":
        prefix="_pth"
    else:
        prefix="_tf"
    folder="Adversarial"+prefix

    load_path=get_checkpoint_name(dataset+"_graph",attack,model_name,folder)
    nbr_bat=len(os.listdir(load_path))
    overall_size=0
    for i in range(nbr_bat):
        dataloader=load_graph_data(dataset,model_name,attack=attack,folder="Adversarial"+prefix,nbr_l_batches=i)
        dataloader = DataLoader(dataloader, batch_size=1, shuffle=True)
        acc_ben=evaluate_GNN(model,dataloader)
        size_data=len(dataloader)
        over_acc=(over_acc*overall_size)+acc_ben*size_data
        overall_size+=size_data
        over_acc=over_acc/overall_size
    folder="Benign"+prefix
    load_path=get_checkpoint_name(dataset+"_graph",None,model_name,folder)
    nbr_bat=len(os.listdir(load_path))
    for i in range(nbr_bat):
        dataloader=load_graph_data(dataset,model_name,attack=None,folder="Benign"+prefix,nbr_l_batches=i)
        dataloader = DataLoader(dataloader, batch_size=1, shuffle=True)
        acc_ben=evaluate_GNN(model,dataloader)
        size_data=len(dataloader)
        over_acc=(over_acc*overall_size)+acc_ben*size_data
        overall_size+=size_data
        over_acc=over_acc/overall_size
    print(f'Accuracy: {over_acc :.2f}%')

if __name__ == "__main__":
    dataset,attack,model_name,folder,task,model_type,model_path,epochs,save,data_size,mode,attr_folder= parseArgs()
    print(f'\n[GRAPH LEARN] dataset: {dataset} | attack: {attack} | model: {model_name} | task: {task} \n')
    model = get_model(model_name,model_type=model_type)
    conv_exis=check_conv_layer(model,model_type)
    if task=="graph":
        
        if conv_exis:
            data_all=load_graph_data(dataset,model_name,attack=None,folder=folder,nbr_l_batches=1)
            data=data_all[0][0]
            model = HGT(hidden_channels=64,metadata=data.metadata(), out_channels=2,
                        num_heads=2,num_layers=2)
            batch_size =1
        else:
            model = GNN_Classifier(1, 20, 2)
            batch_size =10
        opt = torch.optim.Adam(model.parameters())
        history=train_GNN(model,opt,epochs,batch_size,data_size,model_name,model_type,attack,dataset,folder,save=save)
        if len(history[3])>0:
            print("GNN Model saved in ", history[3])
        model=history[0]
        #GNN_eval(model,dataset,model_name,attack,model_type)

    elif task=="GNN_explainer":


        if conv_exis:
            data_all=load_graph_data(dataset,model_name,attack=None,folder=folder,nbr_l_batches=1)
            data=data_all[0][0]
            model = HGT(hidden_channels=64,metadata=data.metadata(), out_channels=2,
                        num_heads=2,num_layers=2)
            gs=DataLoader([data],batch_size=1, shuffle=False)
            for graph_batch in gs:
                model_output(model,graph_batch,"hetero")   
        else:
            model = GNN_Classifier(1, 20, 2)
        model.load_state_dict(torch.load(model_path))
        model_path=model_path.split("models/")[1]
        model=[model,model_path]
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_num=0
            model[0]=model[0].cuda(gpu_num)
        save_path=get_checkpoint_name(dataset+"_graph",attack,model_name,folder)
        nbr_batch=len(os.listdir(save_path))
        get_attributes (dataset,model,model_name,nbr_batch,mode,task,attack=attack,folder=folder,conv_exis=conv_exis,root_folder=attr_folder)
 
    else:
        test_benign_accessor = Accessor('./Benign/'+dataset+'/' +model_name +'/')
        test_adv_accessor= Accessor('./Adversarial/'+dataset+'/'+attack +'/' +model_name +'/' )
        train_benign_accessor = Accessor('./Ground_Truth/'+dataset+'/' +model_name +'/')
        train_adv_accessor = Accessor('./Ground_Truth/'+dataset+'/FGSM/' +model_name +'/')
        
        print('Loading Benign training activations...')
        train_benign_act = train_benign_accessor.get_all()#limit=100)
        print('Loading Adversarial training activations...')
        train_adv_act = train_adv_accessor.get_all()#limit=100)
        print('Loading Benign testing activations...')
        test_benign_act = test_benign_accessor.get_all()#limit=10)
        print('Loading Adversarial testing activations...')
        test_adv_act = test_adv_accessor.get_all()#limit=10)
        #gt_sample_act = ground_truth_accessor.get_all()


        # Transforms the activations to the folowing data set : x[activationA,activaitonB,...]  y= [1, 0 ,1...]
        X_adv_train,Y_adv_train = adversarial_detection_set(train_adv_act,label = torch.tensor(1.0))
        X_ben_train,Y_ben_train = adversarial_detection_set(train_benign_act,label = torch.tensor(0.0))
        X_adv_test,Y_adv_test = adversarial_detection_set(test_adv_act,label = torch.tensor(1.0))
        X_ben_test,Y_ben_test = adversarial_detection_set(test_benign_act,label = torch.tensor(0.0))
        #X_gt ,Y_gt =adversarial_detection_set(gt_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)
        
        print('Shape of training adv activations:',X_adv_train.shape)
        print('Shape of training ben activations:',X_ben_train.shape)
        print('Shape of testing adv activations:',X_adv_test.shape)
        print('Shape of testing ben activations:',X_ben_test.shape)
        
        print('We sample equal number of adversarial and benign data ...')
        
        shape_min = np.min([X_adv_train.shape[0],X_ben_train.shape[0]])
        X_train = torch.cat((X_adv_train[:shape_min],X_ben_train[:shape_min]))# + X_gt
        Y_train= torch.cat((Y_adv_train[:shape_min], Y_ben_train[:shape_min]))# + Y_gt
        
        shape_min = np.min([X_adv_test.shape[0],X_ben_test.shape[0]])
        X_test = torch.cat((X_adv_test[:shape_min],X_ben_test[:shape_min]))# + X_gt
        Y_test= torch.cat((Y_adv_test[:shape_min], Y_ben_test[:shape_min]))# + Y_gt
        #print('Shape of all activations:',X.shape)
        print(f'[GRAPH LEARNING]  Training data : X_train {len(X_train)} |  Y_train {len(Y_train)}')
        print(f'[GRAPH LEARNING]  Testing data : X_test {len(X_test)} |  Y_test {len(Y_test)}')
        
        train_on_activations(X_train,Y_train,X_test,Y_test,model_name,model_path)
