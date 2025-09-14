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
from library.actions_selections import test_robustness
from library.utils import get_dataset,generate_attack, printProgressBar,gen_graph_data,get_checkpoint_name,get_activations_pth
from library.Activations import Activations
from library.train import evaluate,evaluate_model,get_model
from torch_geometric.loader import DataLoader
import os
# ---- silence Python warnings ----
import warnings
warnings.filterwarnings("ignore")  # ignore ALL warnings

import torch
import numpy as np
import sys
from batchup import data_source
import pandas as pd
from keras import backend as K
#from keras.utils import to_categorical
import tensorflow as tf
from argparse import ArgumentParser
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
get_dataset() in utils would return a dataset of your choice, 
Datasets implemented are mnist , cifar , ember and cuckoo,
the function will returna partitioned (X_train, y_train), (X_test, y_test) set for training and testing set
Other params of this function are  Normalization and to_categorical which will be executed if their flag is set to True



get_model() will retreive a model of your choice, the models are in ./models folder
pre-implemented models are mnist_1 , mnist_2,mnist_3,Ember_2 , cifar10 and cuckoo,
get model takes as input a string containing the name of the model

generate_attack_tf() : contains all attacks supported by this software
FGSM, PGD fro cifra10 and mnist
CKO for cuckoo
and EMBER for ember dataset

input :( model:tf.model , X:features , Y: labels , s : attack_name)


generate_activations(): 
will compute and save the activations of a dataset, it takes as input : 
(X : features, Y:labels, model: tf.model, s : file _path to save activations in )


utils.py containes some helpers function that could help you with other functionalities such as 
evaluating the models
'''

supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks =  ['FGSM','CW','PGD',"Bit-Flip","BIM",'Square','APGD-CE','APGD-DLR','Emb-Att',"SPSA","HSJA","SIT",None]
pre_trained_models = ['cifar10_2','cuckoo_1','ember_1','mnist_1','mnist_2','mnist_3']
folders = ['Ground_Truth_tf' , 'Benign_tf' ,'Adversarial_tf','Ground_Truth_pth',"test", 'Benign_pth' ,'Adversarial_pth']
tasks=["graph","default"]
model_types=["keras","pytorch"]
def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', default="mnist", type=str,help=f'supported_dataset= {supported_dataset}')
    parser.add_argument('-model_name', dest='model_name', default="mnist_1", type=str,help=f'pre_trained_models= {pre_trained_models}')
    parser.add_argument('-folder', dest='folder', default="Benign_pth", type=str,help=f'folders= {folders}')
    parser.add_argument('-attack', dest='attack', default=None,help=f'supported_attacks= {supported_attacks}')
    parser.add_argument('-task', dest='task', default="default",type=str,help=f'tasks= {tasks}')
    parser.add_argument('-model_type', dest='model_type', default="pytorch",type=str,help=f'model_types= {model_types}')
    parser.add_argument('-stop', dest='stop', default=0,type=int,help=f'number of batchs to generate for the graph 0 means genrate all possible batches')
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
    return dataset,attack,model_name,folder,task,model_type,args.stop

 




def select_only_one_label (X,Y,label):
    aux = []
    for m,n in enumerate(Y):
        if(np.argmax(n)== label):
            aux.append(X[m])
    return np.array(aux)

def is_wrong_prediction (model,x,y,i):
    pred = np.argmax(model.predict(x,verbose =0))
    return pred == y


# return the activations of each mlayer of the mdel  
def get_layers_activations(model,x):
    inp = model.input                                           
    outputs = [layer.output for layer in model.layers]          
    functors = [K.function([inp], [out]) for out in outputs]  
    layer_outs = [func([x]) for func in functors]
    return layer_outs

def generate_activations(X,Y,model,model_type,file_path):
    print(torch.Tensor(Y).unique())
    counter = 1
    correct_predictions = 0
    for i,x in enumerate(X):
        # Catgorized label to label
        if(isinstance(Y, pd.DataFrame)) : y = Y.iloc[i]
        else : y = Y[i]
        x=x.to(device)
        model=model.to(device)

        #Reshape needed for K backend logits extractions
        if model_type=="keras":
            x = np.expand_dims(x,axis= 0)
         
        #generate and save activations nd return if sucessfull prediction or not
        if(generate_and_save_activations(model,x,i,y,file_path,model_type)):
            '''accuracy should be 100% for benign activations and 0% for adversarial ones'''
            correct_predictions+=1
        #break
        #Print the accuracy so far to monitor hidden layer extraction
        if(counter %50 ==0):
            print(f'accuracy so far : {correct_predictions/counter*100} %')

        printProgressBar(i + 1, X.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1
        
    print("Generated and saved set activations dataset to %s " %(file_path))


#Generates acivations for a given model and input and saves in corresponding folder
def generate_and_save_activations(model,x,index,label,folder_name,model_type):
    import time 
     
    T=time.time()
    '''We generate activations on for evasive adversarial samples
    and correctly predicted benign samples'''
    
    #
    #print(ac)
    #print(len(ac))
    
    if model_type=="keras":
        ac =  get_layers_activations(model,x)
    #extracting arrays of each layer's node
        prediction =np.argmax(model.predict(x,verbose=0)[0])
        activations = [item for sublist in ac for item in sublist]
        if(label.shape[0]!= 1):
            label =np.argmax(label)
    else:
        x=x.to(device)
        x_pred=torch.unsqueeze(x, dim=0)
        if x_pred.shape[-1]==2381:
            prediction=(((model(x_pred)).cpu().detach().numpy()) > 0.4).astype(int)
            prediction=prediction[0][0]
        elif (model(x_pred)).cpu().detach().numpy().data.shape[1]==1:
            prediction=(((model(x_pred)).cpu().detach().numpy()) > 0.5).astype(int)
            prediction=prediction[0][0]
        else:
            prediction =np.argmax((model(x_pred)).cpu().detach().numpy())
        activations_pth=get_activations_pth(x, model,task="default",act_thr=0)
        activations=[act.cpu().detach().numpy() for act in activations_pth]
        label=np.array(int(label))
    ## For adversarial activation extraction we are interested only in evasive samples
    #if any(adv in folder_name for adv in ['Adversarial','FGSM','PGD','CW',"square",'APGD-CLR','APGD-CE',"HSJA","SPSA",'CKO','EMBER_att']):
    #    if prediction==label: #skip extraction
    #        #print("skipping extraction for unevasive sample")
    #        return False
    #else:
    #    if prediction!=label: #skip extraction
            #print("skipping extraction for naturally evasive sample")
    #        return True
    #for arr in activations:
    #     print(arr.shape)
    #print(activations)
    #print(len(activations))
    #print(np.array(activations[0][0]).shape)

    #For string label translated to dummy categories need to format label to string to put in file name
    '''
    if(isinstance(label,pd.Series)):
       label = label['Benign'].astype(str) + label['Malware'].astype(str)
    '''
    lst= []
    for i in activations[:-1] :

        arr = np.array(i)
        if(len(arr.shape) ==4):
            if model_type=="keras":
                arr = np.moveaxis(arr, [0,1,2,3], [3,2,1,0])
            else:
                arr = np.moveaxis(arr, [1,2,3,0] , [0,1,2,3])
        if(len(arr.shape)==2):
            arr = np.moveaxis(arr, [0,1], [1,0])
        arr = np.squeeze(arr)
        #print(arr.shape)
        lst.append(arr)
    
    #print(len(lst))
    #print(lst)
    
    #for arr in lst:
    #     print(arr.shape)
    a = Activations(index,label,prediction,lst)
    a.save_cnn(lst,folder_name)

    return label == prediction 


if __name__ == "__main__":
    dim=0
    print("#############################################################################\n")
    print("#############################################################################\n")
    print("#############################################################################\n")
    print("\n")
    print("\n")
    print("IPG Extraction")
    print("\n")
    print("\n")
    print("#############################################################################\n")
    print("#############################################################################\n")
    print("#############################################################################\n")


    dataset,attack,model_name,folder,task,model_type,stop_b = parseArgs()

    if task=="graph":
        save_path = get_checkpoint_name(dataset+"_graph",attack,model_name,folder)
        
    else:

        save_path = get_checkpoint_name(dataset,attack,model_name,folder)
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    print(f'\n[GEN ACT] Dataset : {dataset} | Model : {model_name} | Attack : {attack} | Checkpoint : {save_path} \n')
    #Use the below  function to generate activations for different datasets/attacks
      # Cifar generation Code
    Nbr_graphs={'mnist':100,'cifar10':100,'ember':1000,'cuckoo':1000}

    # Ground Truth -> We Use Train Data 
    # Adersarial | Benign -> We use Test Data
    if model_type=="keras":
        (X_train, y_train), (X_test, y_test) = get_dataset(dataset,True,model_type=model_type,shuffle=False)
        if(folder in ['Ground_Truth_tf']):
            X = X_train 
            Y= y_train 
        else : 
            X= X_test 
            Y = y_test
    else:
        train_loader,test_loader=get_dataset(dataset,False,model_type=model_type,shuffle=False,batch_size=200,attack=attack,model_name=model_name)
        if(folder in ['Ground_Truth_pth']):
            data_loader=train_loader
        else:
            data_loader=test_loader 
    model = get_model(model_name,model_type=model_type)

    if model_type=="keras" :
        dim_sample=X[0].shape
        dim=(-1,)
        for i in dim_sample:
            dim=dim+(i,)
        evaluate(model,X,tf.convert_to_tensor(Y))
    else:
        batch_size=200
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
        model=model.to(device)
        evaluate_model(model,data_loader,device=device)
    
    if(attack):
        if model_type=="keras" : 
            X_adv=X.copy()
            ds = data_source.ArrayDataSource([X, Y])
            i=0
            batch_size=200
            for (batch_X, batch_y) in ds.batch_iterator(batch_size=batch_size):
                print('Generating {} from sample {} to {}'.format(attack,i*batch_size,(i+1)*batch_size-1))
                X_adv[i*batch_size:(i+1)*batch_size] = generate_attack(model,batch_X, batch_y,attack)
                #evaluate(model,X_adv[i*batch_size:(i+1)*batch_size],tf.convert_to_tensor(batch_y))
                i+=1
            X=X_adv
            evaluate(model,X,tf.convert_to_tensor(Y))
        else:
            X_adv=X.clone()
            
            i = 0
            batch_size = 200
            for (batch_X, batch_y) in data_loader:
                print('Generating {} from sample {} to {}'.format(attack, i * batch_size, (i + 1) * batch_size - 1))
                batch_X=batch_X.to(device)
                batch_y=batch_y.to(device)
                adv_batch_X = generate_attack(model, batch_X, batch_y, attack,model_type=model_type)
                adv_batch_X=adv_batch_X.to("cpu")
                X_adv[i * batch_size:(i + 1) * batch_size] = adv_batch_X
                i += 1
                if task=="graph":
                    if ((i + 1) * batch_size)>(3000):
                        X=X_adv
                        break
            
            dataset_loader=[(x,y) for x,y in zip(X[:(i + 1) * batch_size],Y[:(i + 1) * batch_size])]
            dataset_loader= DataLoader(dataset_loader, batch_size=200, shuffle=True)
            evaluate_model(model,dataset_loader,device=device)
    if task=="graph" :
        gen_graph_data(X,Y,model,save_path,model_type=model_type,attack=attack,dim=dim,batch_size=Nbr_graphs[dataset],stop_point=stop_b,nbr_samples_per_class=Nbr_graphs[dataset])
    else:
        generate_activations(X,Y,model,model_type,save_path) 
