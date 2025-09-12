
from tqdm import tqdm
#from cleverhans.tf2.attacks.spsa import spsa
#from pandas import get_dummies
import numpy as np
import networkx as nx
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.datasets import mnist
from keras.datasets import cifar10
import os
from .SIA_attack import SIA

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,)
from torchvision import datasets, transforms
import torch
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import InMemoryDataset, Data,DataLoader
from batchup import data_source
import dgl
from keras.utils import to_categorical
from keras.models import  Model
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent as projected_gradient_descent_tf
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.hop_skip_jump_attack import HopSkipJumpAttack
#from cleverhans.torch.attacks.spsa import spsa
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as fast_gradient_method_tf
import tensorflow as tf
#from pandas import get_dummies
import ember
import pickle
import pandas as pd
import numpy as np 
from autoattack.autopgd_base import APGDAttack
from autoattack.square import SquareAttack
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""The SPSA attack."""
import numpy as np
import torch
from torch import optim
from cleverhans.torch.utils import clip_eta


def spsa(
    model_fn,
    x,
    eps,
    nb_iter,
    norm=np.inf,
    clip_min=-np.inf,
    clip_max=np.inf,
    y=None,
    targeted=False,
    early_stop_loss_threshold=None,
    learning_rate=0.01,
    delta=0.01,
    spsa_samples=128,
    spsa_iters=1,
    is_debug=False,
    sanity_checks=True,
):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which is useful when
    the model is non-differentiable, or more generally, the gradients do not point in useful
    directions.

    :param model_fn: A callable that takes an input tensor and returns the model logits.
    :param x: Input tensor.
    :param eps: The size of the maximum perturbation, measured in the L-infinity norm.
    :param nb_iter: The number of optimization steps.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: If specified, the minimum input value.
    :param clip_max: If specified, the maximum input value.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the
              default, will try to make the label incorrect. Targeted will instead try to
              move in the direction of being more like y.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end as
              soon as the loss is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size
              (the number of evaluated inputs for each update) is `spsa_samples *
              spsa_iters`
    :param spsa_iters:  Number of model evaluations before performing an update, where each
              evaluation is on `spsa_samples` different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    if y is not None and len(x) != len(y):
        raise ValueError(
            "number of inputs {} is different from number of labels {}".format(
                len(x), len(y)
            )
        )
    if y is None:
        y = torch.argmax(torch.Tensor(np.array(model_fn(x).cpu().detach().numpy())), dim=1)
        y=y.to(x.device)
    # The rest of the function doesn't support batches of size greater than 1,
    # so if the batch is bigger we split it up.
    if len(x) != 1:
        adv_x = []
        x=torch.Tensor(x)
        for x_single, y_single in zip(x, y):
            adv_x_single = spsa(
                model_fn=model_fn,
                x=x_single.unsqueeze(0),
                eps=eps,
                nb_iter=nb_iter,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y_single.unsqueeze(0),
                targeted=targeted,
                early_stop_loss_threshold=early_stop_loss_threshold,
                learning_rate=learning_rate,
                delta=delta,
                spsa_samples=spsa_samples,
                spsa_iters=spsa_iters,
                is_debug=is_debug,
                sanity_checks=sanity_checks,
            )
            adv_x.append(adv_x_single)
        return torch.cat(adv_x)

    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x

    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []
    
    
    # If a data range was specified, check that the input was in that range
    asserts.append(torch.all(x >= clip_min))
    asserts.append(torch.all(x <= clip_max))

    if is_debug:
        print("Starting SPSA attack with eps = {}".format(eps))

    perturbation = (torch.rand_like(x) * 2 - 1) * eps
    _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for i in range(nb_iter):

        def loss_fn(pert):
            """
            Margin logit loss, with correct sign for targeted vs untargeted loss.
            """
            logits = model_fn(x + pert)
            loss_multiplier = 1 if targeted else -1
            return loss_multiplier * _margin_logit_loss(logits, y.expand(len(pert)))

        spsa_grad = _compute_spsa_gradient(
            loss_fn, x, delta=delta, samples=spsa_samples, iters=spsa_iters
        )
        perturbation.grad = spsa_grad
        optimizer.step()

        _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)

        loss = loss_fn(perturbation).item()
        if is_debug:
            print("Iteration {}: loss = {}".format(i, loss))
        if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:
            break

    adv_x = torch.clamp((x + perturbation).detach(), clip_min, clip_max)

    if norm == np.inf:
        asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
    else:
        asserts.append(
            torch.all(
                torch.abs(
                    torch.renorm(adv_x - x, p=norm, dim=0, maxnorm=eps) - (adv_x - x)
                )
                < 1e-6
            )
        )
    asserts.append(torch.all(adv_x >= clip_min))
    asserts.append(torch.all(adv_x <= clip_max))

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def _project_perturbation(
    perturbation, norm, epsilon, input_image, clip_min=-np.inf, clip_max=np.inf
):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = clip_eta(perturbation, norm, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation, clip_min, clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())
    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        with torch.no_grad():
            loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = (
            torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta
        )
        grad_list.append(avg_grad)

    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)
def load_attri(dataset,model_name,folder,mode,attack,all_nodes,model_path,nb_nodes,attri_folder="/data/attributions_data/"):
    nodes_weights=[[] for i in range(len(all_nodes))]
    nodes_act=[[] for i in range(len(all_nodes))]
    if attack:
        id="_attributes_"+attack
    else:
        id="_attributes_ben"
    count=0
    samples_attributes=[]
    batch_path=get_checkpoint_name(dataset+"_graph",attack,model_name,folder)
    nbr_batch=len(os.listdir(batch_path))
    save_path=attri_folder+mode+"/"+model_path
    for nbr in range(nbr_batch):
        data_ben=load_graph_data(dataset,model_name,attack=attack,folder=folder,nbr_l_batches=nbr)
        with open(save_path+'/batch_'+str(nbr)+id+'.pickle', 'rb') as handle:
            samples_attributes = pickle.load(handle)
        for j in tqdm(range(len(data_ben))):
            hetero_explanation=samples_attributes[j]
            nodes=data_ben[j][2]
            stat=len(hetero_explanation.node_mask)==len(nodes)
            if stat:
                for i in range(nb_nodes):
                    if all_nodes[i] in nodes.keys():
                        nodes_weights[i].append(float(hetero_explanation.node_mask[int(nodes[all_nodes[i]])].mean()))
                        nodes_act[i].append(float(data_ben[j][0].x[int(nodes[all_nodes[i]])]))

                    else:
                        nodes_act[i].append(0)
                        nodes_weights[i].append(0)

            else :
                count+=1
    return nodes_weights,nodes_act

def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.

    The loss is high when `label` is unlikely (targeted by default).
    """

    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float("-inf")),
        logits,
    )
    max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

    return max_incorrect_logits - correct_logits
import foolbox as fb
import random

from sklearn import model_selection
def get_checkpoint_name(dataset,attack,model_name,folder):
    # Generate FoLder Path 
    save_path = folder +'/' +dataset +'/'
    save_path += model_name+'/' 
    if(attack):
        save_path+= attack +'/' 
    else:
        save_path+= "benign" +'/'
    return save_path
def generate_attack(model,x,y,attack,model_type="keras"):

    if(attack not in ['FGSM','CW','PGD',"Bit-Flip","BIM",'Square','APGD-CE','APGD-DLR','Emb-Att',"SPSA","HSJA","SIT"]):
        raise Exception("Attack not supported")
        
    #The attack requires the model to ouput the logits
    if(attack== 'FGSM'):
        if model_type=="keras":
            logits_model = tf.keras.Model(model.input,model.layers[-1].output)
            x_adv =  fast_gradient_method_tf(logits_model,x,0.3,norm=np.inf,targeted=False)
        else:
            x_adv=   fast_gradient_method(model,x,0.3,norm=np.inf,targeted=False)
    if (attack== 'BIM'):
        x=x.to("cuda:0")
        y=torch.tensor(y,dtype=torch.long,device=x.device)
        model = model.to("cuda:0")
        fmodel = fb.PyTorchModel(model, bounds=(x.min(), x.max()))
        attack = fb.attacks.LinfBasicIterativeAttack()
        epsilons = [0.3]
        _, advs, success = attack(fmodel, x, y, epsilons=epsilons)
        x_adv=advs[0]
    if(attack=='CW'):
        x_adv = carlini_wagner_l2(model,x,targeted=False)
    if(attack=='HSJA'):
        x_adv = hop_skip_jump_attack(model,x,np.inf,verbose=False,num_iterations=4)
    if(attack=='SIT'):
        attacker = SIA(model)
        if y.dtype != torch.long:
            y = y.long()
        x_adv = attacker(x,y)
        x_adv=x_adv+x
    if(attack=='SPSA'):
        x_adv = spsa(model,x,y=None,eps=0.3,nb_iter=100,norm = np.inf,sanity_checks=False)
    if(attack=='APGD-CE'):
        adversary = APGDAttack(model, norm='Linf', eps=0.3)
        x_adv = adversary.perturb(x)
    if(attack=='APGD-DLR'):
        adversary = APGDAttack(model, norm='Linf',loss="dlr", eps=0.3,n_iter=40,rho=0.01)
        x_adv = adversary.perturb(x)
    if(attack=='Square'):
        adversary = SquareAttack(model, norm='Linf', eps=0.3)
        x_adv = adversary.perturb(x)
    if(attack=='PGD'):
        if model_type=="keras":
            x_adv = projected_gradient_descent_tf(model, x, 0.3, 0.01, 40, np.inf)
        else:
            x_adv = projected_gradient_descent(model,  x, 0.3, 0.01, 40, np.inf)
    if(attack =='Bit-Flip'):
        if model_type=="keras":
            adv = []
            for s in x :
                adv.append(reverse_bit_attack(s,500))
            x = np.array(adv)
        else:
            adv = []
            device=x.device
            x=x.cpu()
            for s in x :
                adv.append(reverse_bit_attack(s,500).detach().numpy())
            x_adv = np.array(adv)
            x_adv = torch.tensor(x_adv, dtype=torch.float32)
    if(attack == "Emb-Att"):
        with open('./data/attack_ember.pickle', 'rb') as handle:
            attack_ember= pickle.load(handle)
        adv = []
        with open('./data/scaleremb.pickle', 'rb') as handle:
            scaler= pickle.load(handle)
        X_adv=x.clone()
        for j,x in tqdm(enumerate(X_adv)):
            for i,s in enumerate(x):
                if not(str(type(attack_ember[i]))=="<class 'list'>") and  len(attack_ember[i].shape)>0 :
                    if attack_ember[i].shape[0]>1:
                        l=list(attack_ember[i].detach().numpy())
                        if s in l:
                            l.remove(s)
                        b=np.random.choice(l)
                        X_adv[j][i]=torch.tensor(b)
                    else:
                        X_adv[j][i]=1.0
                else:
                    random_float = random.uniform(attack_ember[i][0], attack_ember[i][1])
                    X_adv[j][i]=random_float
                labels=y[j]
                x_inter=torch.tensor(scaler.transform(X_adv[j].reshape((1,X_adv[j].shape[0])).cpu().detach().numpy())).to(x.device)
                model.eval()                     
                predicted=(model(x_inter) > 0.5).float()
                predicted=predicted.view(-1).cpu().numpy()
                labels=labels.view(-1).cpu().numpy()
                correct = (predicted == labels).sum().item()
                if correct != 1:
                    break
        xx=torch.tensor(scaler.transform(X_adv.cpu().detach().numpy()))
        x_adv=xx
    return x_adv

    

def attack_Ember(x):
    aux = []
    for i in x :
        k=i
        if(i>10) :
            k +=  (i/100)*2
        aux.append(k)
    return np.array(aux)

# Returns dataset and applies transformation according to parameters
def get_dataset(dataset_name, categorical=False, model_type='keras',batch_size=64,loader="True",shuffle=True,attack=None,model_name=""):
    

    if(dataset_name == "mnist"):
        if model_type=='pytorch':
            # Define a transform to normalize the data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            # Download and load the training data
            trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

            # Download and load the test data
            testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
        elif model_type=='keras':
            (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train / 255.0
            X_test = X_test/ 255.0

    if(dataset_name == "cifar10"):
        if model_type=='keras':
            (X_train, Y_train), (X_test, Y_test)  = cifar10.load_data()
            X_train = X_train.reshape((X_train.shape[0],32,32,3))
            X_test = X_test.reshape((X_test.shape[0],32,32,3))
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train / 255.0
            X_test = X_test/ 255.0
        if model_type=='pytorch':
            if model_name=="cifar10_2" or model_name=="cifar10_6" or model_name=="cifar10_5":
                transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                trainset = datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform_train)
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

                testset = datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, shuffle=False, num_workers=2)
            else:

                # Transformations - Convert images to PyTorch tensors and normalize them
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                # Load the training and test sets
                trainset = datasets.CIFAR10(root='./cifar10', train=True,
                                                        download=True, transform=transform)
                train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

                testset = datasets.CIFAR10(root='./cifar10', train=False,
                                                    download=True, transform=transform)
                test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    if(dataset_name =="cuckoo"):
        if model_type=="keras":
            df = pd.read_csv("./data/cuckoo.csv",encoding='iso-8859-1')
            df.drop(['Samples'],axis=1,inplace=True) # dropping the name of each row
            df_train=df.drop(['Target'],axis=1)
            df_train.fillna(0)
            df['Target'].fillna('Benign',inplace = True)    
            X= df_train.values
            Y=df['Target'].values
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=7)
        else:
            t_loa,tes_loa,tr_set,tes_set=load_cuckoo(batch_size)
            return((t_loa,tes_loa) if loader else (tr_set,tes_set) )
    if(dataset_name == 'ember'):
            t_loa,tes_loa,tr_set,tes_set=load_ember(batch_size,attack)
            return((t_loa,tes_loa) if loader else (tr_set,tes_set) )
    if categorical:
        if(dataset_name == 'cuckoo'):
            Y_train = pd.get_dummies(Y_train)
            Y_test = pd.get_dummies(Y_test)
        else : 
            Y_train= to_categorical(Y_train)
            Y_test= to_categorical(Y_test)
    if(model_type=='keras'):
        return(X_train, Y_train), (X_test, Y_test)
    elif (model_type=='pytorch'):
        return ((train_loader, test_loader) if loader else  (trainset, testset))
    
  
def reverse_bit_attack(x,Nb):
    
    '''Generating adversarial samples by randomly flipping 
    Nb bits from 0 to 1
    For cuckoo dataset
    ''' 
    if "Tensor" in str(type(x)):

        X_test_crafted=x.clone()
    else :
        X_test_crafted=x.copy()
    N_flipped=0
    for index in range(len(x)):
        if x[index] == 0:
            X_test_crafted[index]=1
            N_flipped+=1
        if N_flipped == Nb:
            break
  
    return X_test_crafted 



    # Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

  

# returns folder name given attack and dataset  
def get_folder_name(attack,dataset):

    if(attack):
        folder = "./adversarial/"+dataset
    else: 
        folder = "./begnign/"+dataset

    # Create the folder if it doeasnt exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if(attack):
        folder += "/"+attack
    
    # Create the folder if it doeasnt exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    return folder
'''
def compute_accuracy_tf(model,X_dataset,Y_dataset):
    from batchup import data_source
    correct = 0
    ds = data_source.ArrayDataSource([X_dataset,Y_dataset])
    
    batch_size=200
    for x, y in ds.batch_iterator(batch_size=batch_size):
    #for i,x in enumerate(X_dataset):
        x= x.reshape(-1,28,28)
        #x= generate_attack_tf(model,x)
        pred = np.argmax(model.predict(x,verbose=0))
        if(pred == y.argmax()):
            correct+=1
    print(correct/Y_dataset.shape[0]*100)
 ''' 


    
def get_shape(d):
    if(d=='mnist') : 
        return (28,28)
    if(d=='cifar10'):
        return (-1,3,32,32)
    
def dispersation_index(x):
    if(len(x)==0): raise ValueError('Dispersaiton of an empty array')
    if(type(x) == torch.Tensor) : return torch.var(x)* torch.var(x) / torch.mean(x)
    return np.var(x) * np.var(x) /np.mean(x) 


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def discretize (arr,nb_bins):
    hist, bins = np.histogram(arr, bins=nb_bins)
    discretized = np.digitize(arr, bins)
    return discretized
   
def scotts_rule(data):
    n = len(data)
    if n == 0:
        return 0
    sigma = np.std(data)
    bin_width = 3.5 * sigma / (n ** (1/3))
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    return num_bins


def plotAcrossPredictions(gt, metric, ben=None,adv=None,Pred_range=10,data='mnist',attack="Ember-Att"):#,data='mnist'): 
    X = np.arange(Pred_range)
 
    
    if(adv) :
        if len(adv)==2:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'FGSM',color ="red")
            plt.bar(X , adv[1], 0.2, label = 'PGD',color ="orange")
            
            plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            if(ben):
                plt.bar(X+0.4 , ben, 0.2, label = 'Benign',color="green")
        else:
            plt.bar(X + 0.2, adv[0], 0.2, label = attack,color ="red")
            plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
            if(ben):
                plt.bar(X , ben, 0.2, label = 'Benign',color="green")

    plt.xticks(X)
    plt.xlabel("Prediction")
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not (os.path.exists('./Results/'+data+'/')):
        os.makedirs('./Results/'+data+'/')
    plt.savefig('./Results/'+data+'/'+metric+'.pdf')
    plt.show()
    
def plotAcrossNodes(gt, metric, ben,adv,Node_range=10,data='mnist',label=0,masking=False,dist=False):#,data='mnist'): 
    plt.figure(figsize=(50,20))
    X = np.arange(Node_range)
    mask=[]
    
    if masking:
        for i in range(Node_range):
            if gt[i]<masking or ben[i]<masking or adv[0][i]<masking or adv[1][i]<masking:
                mask.append(False)
            else:
                mask.append(True)
    else:
        mask=[True]*Node_range
            
    X=X[mask]
    gt=gt[mask]
    adv[0]=adv[0][mask]
    adv[1]=adv[1][mask]
    ben=ben[mask]        
    
    print('Number of Nodes: ',len(X))
    
    if dist:
        if len(adv)==2:
            plt.bar(X - 0.1, abs(adv[0] - gt) , 0.2, label = 'FGSM - GT',color ="red")
            plt.bar(X , abs(adv[1] - gt), 0.2 , label = 'PGD - GT',color ="orange")
            
            #plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            
            plt.bar(X+0.1 , ben, 0.2, label = 'Benign- GT',color="green")
        else:
            plt.bar(X - 0.1, abs(adv-gt), 0.2, label = 'adv - GT',color ="red")
            #plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
            
            plt.bar(X , abs(ben-gt), 0.2, label = 'Benign - GT',color="green")
    else:
        if len(adv)==2:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'FGSM',color ="red")
            plt.bar(X , adv[1], 0.2, label = 'PplotAcrossPredictionsGD',color ="orange")
            
            plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            plt.bar(X+0.4 , ben, 0.2, label = 'Benign',color="green")
        else:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'adv',color ="red")
            plt.bar(X - 0.2, gt, 0.2, label = 'plotAcrossPredictionsGroundTruth',color="grey")
            plt.bar(X , ben, 0.2, label = 'Benign',color="green")

    plt.xticks(X,rotation = 90,fontsize=5)
    plt.xlabel("Nodes")
    plt.ylabel(metric)
    #ymin = np.min([np.min(adv[0]),np.min(adv[1]),np.min(ben), np.min(gt)])
    #ymax = np.max([np.max(adv[0]),np.max(adv[1]),np.max(ben),np.max(gt)])
    #print(ymin, ymax)
    #plt.ylim([ymin,ymax])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not (os.path.exists('./Results/'+data+'/'+metric)):
        os.makedirs('./Results/'+data+'/'+metric)
    plt.savefig('./Results/'+data+'/'+metric+'/pred='+str(label)+'.pdf')
    plt.show()
    
    
def plotAcrossLayers(gt, metric, ben,adv,layer_range=10,data='mnist',label=0,masking=False,dist=False):#,data='mnist'): 
    #plt.figure(figsize=(50,20))
    X = np.arange(layer_range)
    mask=[]
    
    if masking:
        for i in range(layer_range):
            if gt[i]<masking or ben[i]<masking or adv[0][i]<masking or adv[1][i]<masking:
                mask.append(False)
            else:
                mask.append(True)
    else:
        mask=[True]*layer_range
            
    X=X[mask]
    gt=gt[mask]
    adv[0]=adv[0][mask]
    adv[1]=adv[1][mask]
    ben=ben[mask]        
    
    print('Number of Layers: ',len(X))
    
    if dist:
        if len(adv)==2:
            plt.bar(X - 0.1, abs(adv[0] - gt) , 0.2, label = 'FGSM - GT',color ="red")
            plt.bar(X , abs(adv[1] - gt), 0.2 , label = 'PGD - GT',color ="orange")
            
            #plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            
            plt.bar(X+0.1 , ben, 0.2, label = 'Benign- GT',color="green")
        else:
            plt.bar(X - 0.1, abs(adv-gt), 0.2, label = 'adv - GT',color ="red")
            #plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
            
            plt.bar(X , abs(ben-gt), 0.2, label = 'Benign - GT',color="green")
    else:
        if len(adv)==2:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'FGSM',color ="red")
            plt.bar(X , adv[1], 0.2, label = 'PGD',color ="orange")
            
            plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            plt.bar(X+0.4 , ben, 0.2, label = 'Benign',color="green")
        else:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'adv',color ="red")
            plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
            plt.bar(X , ben, 0.2, label = 'Benign',color="green")

    plt.xticks(X)#,rotation = 90,fontsize=5)
    plt.xlabel("Layers")
    plt.ylabel(metric)
    #ymin = np.min([np.min(adv[0]),np.min(adv[1]),np.min(ben), np.min(gt)])
    #ymax = np.max([np.max(adv[0]),np.max(adv[1]),np.max(ben),np.max(gt)])
    #print(ymin, ymax)
    #plt.ylim([ymin,ymax])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not (os.path.exists('./Results/'+data+'/'+metric)):
        os.makedirs('./Results/'+data+'/'+metric)
    plt.savefig('./Results/'+data+'/'+metric+'/pred='+str(label)+'.pdf')
    plt.show()
    
def plotDiff(FGSM_diff,PGD_diff,adv_diff,Node_range=10,data='mnist',label=0):#,data='mnist'): 
    plt.figure(figsize=(50,20))
    X = np.arange(Node_range)
    
    
    
    plt.bar(X - 0.1, FGSM_diff , 0.2, label = 'FGSM - Ben',color ="red")
    plt.bar(X , PGD_diff, 0.2 , label = 'PGD - Ben',color ="orange")
    
    #plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
    
    plt.bar(X+0.1 , adv_diff, 0.2, label = 'FGSM- PGD',color="blue")
        

    plt.xticks(X,rotation = 90,fontsize=5)
    plt.xlabel("Nodes",fontsize=10)
    plt.ylabel('Average Activation Difference',fontsize=10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not (os.path.exists('./Results/'+data+'/Diffirence_in_activations/')):
        os.makedirs('./Results/'+data+'/Diffirence_in_activations/')
    plt.savefig('./Results/'+data+'/Diffirence_in_activations/pred='+str(label)+'.pdf')
    plt.show()



class MyDGLDataset:
    def __init__(self, graph_list, labels_list):
        self.graphs = graph_list
        self.labels = labels_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
# this function loads benign and adversial dataset , suffle and return them for the training by batch
def gen_train_batch(dataset,model_name,attack,nb_bt,folder):
    graph_list = []
    labels_list = []
    gnn_loader=load_graph_data(dataset,model_name,attack=None,folder=folder,nbr_l_batches=nb_bt)
    #gnn_loader = DataLoader(gnn_loader, batch_size=1, shuffle=False)

    for batche,label,nodes,y_class,pred_state in gnn_loader:
            labels_list.append(label)
            graph_list.append(batche)
    gnn_loader=load_graph_data(dataset,model_name,attack=attack,folder=folder,nbr_l_batches=nb_bt)
    #gnn_loader = DataLoader(gnn_loader, batch_size=1, shuffle=False)
    for batche,label,nodes,y_class,pred_state in gnn_loader:
            labels_list.append(label)
            graph_list.append(batche)
    
    gnn_loader=0
    dataset = MyDGLDataset(graph_list, labels_list)
    return(dataset)
def load_graph_data(dataset,model_name,attack=None,folder="",nbr_l_batches=1):
    load_path=get_checkpoint_name(dataset+"_graph",attack,model_name,folder)    
    dataset= DynamicGraphDataset(root=load_path+"/batch_"+str(nbr_l_batches))
    dataset.load()
    return(dataset)
def check_conv_layer(model,model_type):
    conv_exist=False
    if model_type=="keras":
        layers=[layer.name for layer in model.layers]
        for i in layers:
            if "conv" in i :
                conv_exist=True
    elif model_type=="pytorch":
        for i in model._modules:
            if "conv" in i :
                conv_exist=True
    return conv_exist
def check_balance(balance_class,y,nbr_samples_per_class):
    return(balance_class[int(y)]<nbr_samples_per_class)



#ge gen_graph_data generate the graph extracted from the model after predicting an input
#stop point is the number of how many batch we want to store
def gen_graph_data(X,Y,model,save_path,model_type="pytorch",dim="",attack=None,batch_size=1000,stop_point=0,nbr_samples_per_class=1000):
    conv_exist=check_conv_layer(model,model_type)
    if model_type=="keras":
        pred=Y.copy()
        ds = data_source.ArrayDataSource([X, Y])
        i=0
        batch_s=200
        for (batch_X, batch_y) in ds.batch_iterator(batch_size=batch_s):
            pred[i*batch_s:(i+1)*batch_s] = model.predict(batch_X,verbose=False)
            #evaluate(model,X_adv[i*batch_size:(i+1)*batch_size],tf.convert_to_tensor(batch_y))
            i+=1
        argmax_indices = np.argmax(pred, axis=1)
        # Create an array of zeros with the same shape as the input array
        output_array = np.zeros_like(pred)
        # Set the corresponding element in each row to 1 based on the argmax index
        for i in range(pred.shape[0]):
            output_array[i, argmax_indices[i]] = 1
        y_0=[]
        target_0=[]
        comparison_array = np.equal(output_array, Y).astype(int)
        if attack:
            y=[0,1]
            for i in range(len(comparison_array)):
                if sum(comparison_array[i])<10 :
                    y_0.append(X[i])
                    target_0.append(Y[i])
        else:
            for i in range(len(comparison_array)):
                if sum(comparison_array[i])==10 :
                    y_0.append(X[i])
                    target_0.append(Y[i])

            y=[1,0]
        X=np.array(y_0)
        Y=np.array(target_0)
    elif model_type=="pytorch":
        X_filtred=[]
        accurate_pred=[]
        Y_filtred=[]
        layer=list( model.children())[-1]
        out_dim=len(Y.unique())
        balance_class={}
        for i in range(out_dim):
            balance_class[i]=0
        for i,x in enumerate(X):
            x=torch.unsqueeze(x, dim=0)
            x=x.to(device)
            model=model.to(device)
            prediction =np.argmax((model(x)).cpu().detach().numpy())
            #print(Y.shape)
            label=np.array(int(Y[i]))
            if attack and check_balance(balance_class,Y[i],nbr_samples_per_class):
                y=[0,1]
                X_filtred.append(X[i])
                Y_filtred.append(Y[i])
                accurate_pred.append(int(prediction!=label))
                balance_class[int(Y[i])]+=1
            elif check_balance(balance_class,Y[i],nbr_samples_per_class):
                X_filtred.append(X[i])
                Y_filtred.append(Y[i])
                balance_class[int(Y[i])]+=1
                accurate_pred.append(int(prediction==label))
                prediction!=label
                y=[1,0]
        X=torch.stack(X_filtred)
        Y=torch.stack(Y_filtred)
    print(balance_class)
    existing_batch=os.listdir(save_path)
    last_nbr=0
    if len(existing_batch)>0:
        last_nbr=max([int(existing_batch[i][-1]) for i in range(len(existing_batch))])+1
    batch_nbr=len(Y)//batch_size
    batch_rest=len(Y)%batch_size
    if stop_point>0:
        batch_nbr=stop_point
    for nbr in  tqdm(range(last_nbr,batch_nbr)):
        dataset= DynamicGraphDataset(root=save_path+"/batch_"+str(nbr))
        for i in tqdm(range(batch_size)):
            g,dic_nodes=get_activations_pth(X[i+(nbr*batch_size)], model,dim=dim,mode="saving_graph",model_type=model_type, conv_exist=conv_exist)
            dataset.add_graph(g, torch.Tensor(y).float(),dic_nodes,Y[i+(nbr*batch_size)],accurate_pred[i+(nbr*batch_size)])
        dataset.save()
        print("Batch number ", nbr," generated and saved in", save_path)
    dataset=0
    existing_batch=os.listdir(save_path)
    if len(existing_batch)>0:
        last_nbr=max([int(existing_batch[i][-1]) for i in range(len(existing_batch))])+1
    if last_nbr==batch_nbr and stop_point==0 :
        dataset= DynamicGraphDataset(root=save_path+"/batch_"+str(batch_nbr))
        for i in tqdm(range(batch_rest)):
            g,dic_nodes=get_activations_pth(X[i+(batch_nbr*batch_size)], model,dim=dim,model_type=model_type,mode="saving_graph", conv_exist=conv_exist)
            dataset.add_graph(g, torch.Tensor(y).float(),dic_nodes,Y[i+(batch_nbr*batch_size)],accurate_pred[i+(batch_nbr*batch_size)])
        dataset.save()
        print("Batch number ", batch_nbr," generated and saved in", save_path)
    dataset=0
#DynamicGraphDataset is a custom class for graph dataset saving and loading
class DynamicGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DynamicGraphDataset, self).__init__(root, transform, pre_transform)
        self.data_list = []
        self.label_list = []
        self.nodes_list=[]
        self.class_list=[]
        self.pred_state=[]

    def add_graph(self, graph_data, label,nodes,y_class,pred_state):
        self.data_list.append(graph_data)
        self.label_list.append(label)
        self.nodes_list.append(nodes)
        self.class_list.append(torch.Tensor(y_class))
        self.pred_state.append(pred_state)
    def save(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((self.data_list, self.label_list,self.nodes_list,self.class_list,self.pred_state), self.processed_paths[0])
    def load(self):
        data, labels ,nodes,y_class,pred_state= torch.load(self.processed_paths[0])
        self.data_list = data
        self.label_list = labels
        self.nodes_list=nodes
        self.class_list=y_class
        self.pred_state=pred_state

    def process(self):
        pass  # You can implement your custom data processing here if needed.

    @property
    def processed_file_names(self):
        return ["your_processed_data.pt"]  # Specify the name of the processed data file

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx],self.nodes_list[idx],self.class_list[idx],self.pred_state[idx]


# Define a function to convert a DGL graph to a PyTorch Geometric Data object
def dgl_to_pyg(dgl_graph):
    pyg_data = Data()
    
    # Copy node features
    pyg_data.x = dgl_graph.ndata['weight']
    
    # Copy edge indices (assuming an undirected graph)
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    pyg_data.edge_index = edge_index
    
    # Add other necessary attributes, such as labels, if available
    # pyg_data.y = ...

    return pyg_data
def convert_to_pyg(G):
    node_types=[]
    import torch
    from torch_geometric.data import HeteroData
    hetero_data = HeteroData()
    # Dictionary to store node types based on weight dimensions
    node_types = {}
    node_values={}
    # Add nodes with different types and features
    for node_id, node_data in G.nodes(data=True):
        weight = node_data.get('weight', torch.zeros(3))  # Default to zeros if 'weight' not present
        if torch.numel(weight)==1:
            weight=torch.Tensor([weight])
        
        node_type = f'{node_id.split(",")[0]}_{len(weight)}'
        if not(node_type in node_values.keys()):
            node_values[node_type]=[]
        node_values[node_type].append(weight)
        # Store the node type for later use
        node_types[node_id] = [node_type]
    for i in list(node_values.keys()):
        hetero_data[i].x=torch.stack(node_values[i], dim=0)
    node_names=[i for i in node_values.keys()]
    edge_ind=0
    node_ind=0
    for i in range(len(node_names)-1):
        src_len=hetero_data[node_names[i]].x.shape[0]
        dst_len=hetero_data[node_names[i+1]].x.shape[0]
        srcs=[]
        dsts=[]
        #id-node_ind for id in src[edge_ind:edge_ind+src_len*dst_len]
        for c_src in range(src_len):
            for c_dst  in range(dst_len):
                srcs.append(c_src)
                dsts.append(c_dst)
        #dsts=[id-node_ind-src_len for id in dst[edge_ind:edge_ind+src_len*dst_len]]
        if i==0:
            hetero_data[node_names[i],"default",node_names[i]].edge_index=torch.tensor([[count for count in range(1)] ,[count for count in range(1)]], dtype=torch.long)
            #hetero_data[node_names[i],"default",node_names[i]].edge_attr=torch.ones((src_len,1), dtype=torch.float).requires_grad_(False)
        hetero_data[node_names[i],"default",node_names[i+1]].edge_index=torch.tensor([srcs ,dsts], dtype=torch.long)
        
        #hetero_data[node_names[i],"default",node_names[i+1]].edge_attr=torch.ones((len(srcs),1), dtype=torch.float).requires_grad_(False)
        edge_ind+=src_len*dst_len
        node_ind+=src_len
    return(hetero_data)

# get_activations_pth amis to construct a networkx graph for the model input and return sub or full graph 
# based on the mode [dgl-sub: dgl subgraph with only active nodes, all_nodes gives all nodes name
# all_edges gives all edges src and dst node names and sub gives the networkx subgraph]
# This function operate based on the model type and conv layers existance to shift
#to Heterogeneous graph learning 
def get_activations_pth(x, model,dim="",task="graph",model_type="pytorch", mode="sub",conv_exist=False, act_thr=0):
    if model_type=="pytorch":  
        # Prepare the input tensor
        model=model.to(device)
        x=x.to(device)
        activation_list=[]
        layer_names=[]
        def hook_fn(module, input, output):
            layer_name = str(module.__class__.__name__)
            activation_list.append(output.clone().detach().cpu())
            layer_names.append(layer_name)
        hooks = []
        for layer in model.children():
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Ensure the model is in evaluation mode
        model.eval()
        # Perform the forward pass
        x=torch.unsqueeze(x, dim=0)
        output = model(x)
        for hook in hooks:
            hook.remove()
        if model.get_activation_functions():
            ac_func=list(model.get_activation_functions().values())
            
            for i in range(len(activation_list)):
                if ac_func[i]!="None":
                    activation_list[i]=ac_func[i](activation_list[i])

        if task=="default":
            return(activation_list)
        else:
            G = nx.Graph()
            # Get a list of layer names in your PyTorch model
            layer_names = [f'layer_{i}' for i in range(len(model._modules))]
            # Process each layer and extract activations
            for i, (layer_name, activations) in enumerate(zip(layer_names,activation_list)):
                # Create a subgraph for each layer
                layer_G = nx.DiGraph()
                # Define a threshold for activation (you can adjust this)
                activation_threshold = act_thr
                activations=activations[0]
                # Add nodes and edges for each neuron in the layer
                nr_id = 0
                for j in range(activations.shape[0]):
                    activation = activations[j].cpu()
                    if not(torch.numel(activation)==1):
                        activation=torch.Tensor(activation.flatten().tolist())
                    if not conv_exist:
                        activation=torch.Tensor([activation])
                    activated =activation.max()!=activation_threshold
                    layer_G.add_node(f"Layer {i} , Neuron {nr_id}", label=f"Layer {i} , Neuron {nr_id}", activated=activated, weight=activation)
                    nr_id += 1
                # Add the subgraph to the main graph
                G = nx.compose(G, layer_G)
            
    elif model_type=="keras":
        # Load your pretrained model
        #model = load_model("models/mnist_1.h5")
        #input_image = test_input
        input_image=x
        input_image=input_image.reshape(dim)
        G = nx.Graph()

        # Get a list of layer names in your model
        layer_names = [layer.name for layer in model.layers]
        layers = [layer for layer in model.layers]

        # Process each layer and extract activations
        for i, layer_name in enumerate(layer_names):
            # Create a subgraph for each layer
            layer_G = nx.DiGraph()
            # Create a model to extract the activations of this layer
            layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

            # Get activations for the input image
            activations = layer_model.predict(input_image,verbose=False)[0]
            
            # Define a threshold for activation (you can adjust this)
            activation_threshold =0
            # Add nodes and edges for each neuron in the layer
        
            for j in range(activations.shape[-1]):
                activation = activations[..., j]
                if (np.size(activation)==1):
                        fixed_shape = (1,)
                        activation=torch.Tensor(activation.flatten().tolist())
                        activation = np.resize(activation, fixed_shape)
                else:
                    activation = activation.flatten()
                    # Convert the flattened array to a NumPy array
                    activation = np.array(activation)
                activated = np.max(activation) != activation_threshold
                if conv_exist:
                    activation=torch.Tensor(activation)
                #fixed_shape = (dim[1]*dim[2],)
                layer_G.add_node(f"Layer {i}, Neuron {j}", label=f"Layer {i} , Neuron {j}\nActivated {activated}", activated=activated,weight=activation)

            # Add the subgraph to the main graph

            G = nx.compose(G, layer_G)
    if not conv_exist: 
        # Connect nodes within each layer
        for i in range(len(layer_names) - 1):
            current_layer_nodes = [node for node in G.nodes if f"Layer {i} " in G.nodes[node]["label"]]
            next_layer_nodes = [node for node in G.nodes if f"Layer {i + 1} " in G.nodes[node]["label"]]
            for node1 in current_layer_nodes:
                for node2 in next_layer_nodes:
                    G.add_edge(node1, node2)
    
    subgraph = nx.DiGraph()
    for node, data in G.nodes(data=True):
        if data.get('activated') == True:
            subgraph.add_node(node, **data)
    if not conv_exist:      
        for edge in G.edges():
            if edge[0] in subgraph and edge[1] in subgraph:
                subgraph.add_edge(*edge)
    dic_nodes = {}
    counter = 0
    nodes = list(subgraph.nodes())
    for i in nodes:
        dic_nodes[i] = counter
        counter += 1
    if mode=="total_graph":
        return(G)
    elif mode == "sub":
        return subgraph, dic_nodes
    elif mode == "dgl-sub":
        return dgl_to_pyg(dgl.from_networkx(subgraph, node_attrs=["weight"]))
    elif mode=="saving_graph":
        if conv_exist:
            return convert_to_pyg(subgraph),dic_nodes
        else:
            return dgl_to_pyg(dgl.from_networkx(subgraph, node_attrs=["weight"])),dic_nodes
    elif mode == "all_nodes":
        return list(G.nodes())
    elif mode == "all_edges":
        if conv_exist:
            return convert_to_pyg(G)
        else:
            return list(G.edges())
    elif mode == "all_edges":
        return list(G.edges())
def boxplot(x_axis, y_ben, y_adv, x_label, y_label, title, xticks=None):
     
    # set width of box
    boxWidth = 0.25
    fig1, ax1 = plt.subplots(figsize=(30,18))
    
    # set x positions
    x_ben = [x - boxWidth/2 for x in x_axis]
    x_adv = [x + boxWidth/2 for x in x_axis]
    
    
    plt.title(title)
    plt.xlabel(x_label)
    #plt.ylabel(y_label)
    #if xticks != None:
    #    plt.xticks(x_axis,labels=xticks,rotation='vertical')
    

    #print(y_ben[0].shape)
    # Creating plot
    bx1 = ax1.boxplot(y_ben,notch=True,whis=2,positions=x_ben,widths=boxWidth,patch_artist=True,showfliers=False)
    ax1.set_xticks(x_axis,labels=xticks,rotation=90)
    ax1.set_ylabel(y_label+' Benign')
    plt.setp(bx1["boxes"], facecolor='green',label='Benign')
    
    #ax2 = ax1.twinx()
    
    # Creating plot
    bx2 = ax1.boxplot(y_adv,notch=True,whis=2,positions=x_adv,widths=boxWidth,patch_artist=True,showfliers=False)
    ax1.set_xticks(x_axis,labels=xticks,rotation=90)
    ax1.set_ylabel(y_label+' Adversarial')
    plt.setp(bx2["boxes"], facecolor='red',label='Adversarial')
    
    by_label={0: 'Adversarial',
             1: 'Benign'}
    
    # show plot
    plt.show()

def plot_act(values_set2,norm,j):

    # Define the width of each bar and the positions for the bars
    index = np.arange(len(values_set2))

    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

    # Create bar plots for each set of values
    #plt.bar(index, values_set1, bar_width, color='red', label='adv')
    plt.bar(index, values_set2, color='blue', label=norm+' (ben-adv)')

    # Add labels and title
    plt.xlabel('nodes',fontsize =18)
    plt.ylabel('Act distance per node',fontsize =18)
    plt.title(norm+" layer"+str(j),fontsize =18)
    plt.xticks(index, index, rotation=45)

    # Add a legend
    plt.legend(fontsize =18)

    # Display the plot
    plt.tight_layout()
    plt.show()
def plot_attri(values_set1,values_set2,j):
    classes=[i for i in range(len(values_set1))]
    # Define the width of each bar and the positions for the bars
    bar_width = 0.5
    index = np.arange(len(classes))

    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

    # Create bar plots for each set of values
    plt.bar(index, values_set1, bar_width,color='green', label='ben' )
    plt.bar(index + bar_width, values_set2, bar_width, color='red', label='adv')

    # Add labels and title
    plt.xlabel('nodes')
    plt.ylabel('Average attributions nodes')
    plt.xticks(index + bar_width / 2, classes, rotation=45)
    plt.title("Layer "+str(j))
    # Add a legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
from sklearn import model_selection
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def load_cuckoo(batch_size):

    df = pd.read_csv("./data/cuckoo.csv",encoding='iso-8859-1')
    df.drop(['Samples'],axis=1,inplace=True)
    values=df.values
    N=len(values[0])-1 # Number of features
    rng=len(values[:,0]) # Number of samples
    Null_values=[] # collecting null samples
    for i in range(rng):
        if list(values[i,0:-1]) == [0]*N:
            Null_values.append(i)
    list(values[Null_values[0],0:-1]) == [0]*N
    # drop null rows
    df.drop(labels=Null_values,axis=0, inplace=True)
    # shuffling records
    df = df.sample(frac=1).reset_index(drop=True)
    values=df.values
    N=len(values[:,0]) # number of elts in a column (samples)
    rng=len(values[0,:]) # number of columns
    Null_values_col=[] # searshing for columns with less than 2 samples
    for i in range(rng-1):
        if list(values[:,i]).count(1) < 2 :
            Null_values_col.append(i)
    # drop features with less than 2 samples
    col_to_drop = []
    for col in Null_values_col:
        #print(df.columns[col])
        col_to_drop.append(df.columns[col])
    df.drop(labels=col_to_drop,axis=1, inplace=True)
    # checking the result
    values=df.values
    N=len(values[:,0]) # number of elts in a column (samples)
    rng=len(values[0,:]) # number of columns
    #Null_values_col=[] # searshing for columns with less than 2 samples
    nb=0
    for i in range(rng-1):
        if list(values[:,i]).count(1) < 2 :
            nb+=1
    df_train=df.drop(['Target'],axis=1)
    df_train.fillna(0)
    X= df_train.values
    df['Target'].fillna('Benign',inplace = True)
    Y=df['Target'].values
    test_size = 0.2
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    # Convert string labels to numeric
    label_encoder = LabelEncoder()
    y_train_numeric = label_encoder.fit_transform(y_train) 

    X# Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_numeric, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))  # Add an extra dimension to y
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    # Convert X_test and y_test to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(label_encoder.transform(y_test), dtype=torch.float32)

    # Create a DataLoader for test data
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor.unsqueeze(1))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return(train_loader, test_loader,train_dataset,test_dataset)
def load_ember(batch_size, attack, data_size=60000, root="./data/ember2018/",
               balance_per_class=5000, num_workers=2, pin_memory=True):
    import os, pickle
    import numpy as np, torch
    from torch.utils.data import Dataset, DataLoader
    import ember

    # 1) Read (these are memmaps in EMBER; cheap if you avoid full copies)
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(root)

    # 2) Keep only labeled rows via indices (no materialization)
    idx_train_all = np.flatnonzero(y_train != -1)
    idx_test_all  = np.flatnonzero(y_test  != -1)

    # Limit training size early (on indices)
    if data_size is not None:
        idx_train_all = idx_train_all[:data_size]

    # 3) (Optional) balance by indices only (no clones)
    def take_balanced(idx, y, k):
        if not k: return idx
        i0 = idx[y[idx] == 0][:k]
        i1 = idx[y[idx] == 1][:k]
        return np.concatenate([i0, i1])

    idx_train = take_balanced(idx_train_all, y_train, balance_per_class)
    idx_test  = take_balanced(idx_test_all,  y_test,  balance_per_class)

    # 4) Load scaler stats and keep them as float32 (no transform copies)
    mean = scale = None
    if not attack:
        with open('./data/scaleremb.pickle', 'rb') as f:
            scaler = pickle.load(f)
        mean  = np.asarray(getattr(scaler, "mean_",  None), dtype=np.float32)
        scale = np.asarray(getattr(scaler, "scale_", None), dtype=np.float32)

    # 5) Lightweight Dataset that scales per-sample (or per-batch, see note)
    class EmberDS(Dataset):
        def __init__(self, X, y, indices, mean=None, scale=None):
            self.X, self.y, self.idx = X, y, indices.astype(np.int64, copy=False)
            self.mean, self.scale = mean, scale
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = int(self.idx[i])
            # row -> float32 array without huge copies
            x = np.asarray(self.X[j], dtype=np.float32, order='C')
            if self.mean is not None and self.scale is not None:
                x = (x - self.mean) / self.scale
            y = int(self.y[j])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    train_ds = EmberDS(X_train, y_train, idx_train, mean, scale)
    test_ds  = EmberDS(X_test,  y_test,  idx_test,  mean, scale)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader, train_ds, test_ds
