import numpy as np

def sort_by_index(act):
    
    ind = [i.index for i in act]
    sorted_act=[x for _, x in sorted(zip(ind, act))]
    
    #for elt in sorted_act:
    #    print(elt.index)
    
    return sorted_act

def Average_act_weight_per_layer(act):
    
    
    nb_layers = act[0].get_nb_layers()
    print(nb_layers)
    avg=[]
    for i in range(nb_layers):
        avg_i=0
        for elt in act:
            avg_i+=np.mean(elt.activations_set[i])
        avg.append(avg_i/len(act))
        
    #print(avg)
            
    return avg

def Average_act_weight_per_node(act):
    
    
    nb_nodes = act[0].get_nb_nodes()
    
    avg=[]
    for i in range(nb_nodes):
        avg_i=0
        for elt in act:
            avg_i+=elt.flatten()[i]
        avg.append(avg_i/len(act))
        
    #print(avg)
            
    return avg

def avg_act_diff(act1,act2,nb_sample=1000):
    '''This function computes the difference between activation weights of two sets 
    of activations across all samples 'nb_samples'''
    
    
    
    nb_sample=np.min([len(act1),len(act2),nb_sample])
    for i in range(nb_sample):
        #print("sample {} index is:".format(i))
        #print(act1[i].index)
        #print(act2[i].index)
        flat1 = act1[i].flatten()
        flat2 = act2[i].flatten()
        if len(flat1) != len(flat2):
            continue
        if i==0:
            diff=abs(np.subtract(np.array(flat1),np.array(flat2)))
        else:
            diff = diff+abs(np.subtract(np.array(flat1),np.array(flat2)))
    #print(diff)
    return (diff/nb_sample).tolist()

def compare(act1,act2,nb_sample=1000):
    
    '''This function computes the similarity of two sets 
    of activations across all samples 'nb_samples'.'''
    
    avg_sim=0
    nb_nodes=act1[0].get_nb_nodes()
    nb_sample=np.min([len(act1),len(act2),nb_sample])
    for i in range(nb_sample):
        sim=0
        
        #if act1[i].get_activations_set()== act2[i].get_activations_set():
            #print('{}% of the nodes of sample {} gave similar activations'.format(100,i))
        #    avg_sim+=1
        try:
            for j in range(nb_nodes):
                if act1[i].flatten()[j] == act2[i].flatten()[j]:
                    sim+=1
            #print('{}% of the nodes of sample {} gave similar activations'.format(100*sim/nb_nodes,i))
            avg_sim+=sim/nb_nodes
        except IndexError:
            pass
        
    return 100*avg_sim/nb_sample

def  expD(act):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each Prediction
    #Comapred
    threshhold = 0
    
    avg_active_node =[]  
    for y in act:
        y.set_layer_range(1,float('+inf'))
        avg_active_node.append(y.compute_nb_active_nodes(threshhold)) 
    
    return np.average(avg_active_node)
    



def expE(act):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    

    summ = 0
    for x in act:
        summ += x.get_average_weight() 
    
    
    return summ/len(act)


def expF(act):
    # This experiment tres to find nodes that are always active 

    threshold = 0
    
    # Intitialize a list of always active nodes
    
    always_active_nodes = list(range(act[0].get_nb_nodes()))
    
    
    # Look for cases where a node is deactivated and remove it
    for x in act :
        unactive_nodes = []
        for j,y in enumerate(x.flatten()):
            if(type(y) == type([])):continue
            if(abs(y)==threshold):
                unactive_nodes.append(j)
        
        for node in unactive_nodes:
            if node in always_active_nodes:
                always_active_nodes.remove(node)
                
        if(len(always_active_nodes) == 0 ):
            break
        
        
    return always_active_nodes

def expG(act):
    #Purpsoe fo this experiment is to compute the frquency of node sactivations


    threshhold = 0

    nb_nodes_in_model = act[0].get_nb_nodes()
    
    frequency = [0]*nb_nodes_in_model
    #indices=list(range(nb_nodes_in_model))
        


    for i in act:
        #i.set_layer_range(l,l)
        flat = i.flatten()
        if(len(flat) !=nb_nodes_in_model ) : continue
     
        for ind,j in enumerate(flat):
            if(type(j) == type([])): continue

            if( abs(j)>threshhold):
                frequency[ind] +=1 / len(act)
    
    return frequency


def expH(act):
    #Dispersation index

    ind = 0
    for i in act:
        ind += i.dispersation_index()

    
    return ind/len(act)


   
def expI(act):

    #Computes entropy of samples
    entropy = 0
    for i in act:
        entropy += i.compute_entropy() / len(act)
    
    
    
    
    
    return entropy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
# Sample data (replace this with your actual data)

def plot_layers_metrics(datasets,labels,y_label,title):
    colors=["green","red","blue","orange","purple"]
    # Set up the figure and axis
    fig, ax1 = plt.subplots(figsize=(15, 10))
    boxWidth = 0.15  # Adjust the width of the boxes

    # X-axis positions
    x_axis = np.arange(len(datasets[0]))

    # Create box plots in a loop
    positions = x_axis - (len(datasets[0]) // 2) * boxWidth
    handles = []  # List to store legend handles

    for i, dataset in enumerate(datasets):
        bx = ax1.boxplot(
            dataset,
            notch=True,
            whis=2,
            positions=positions + i * boxWidth,
            widths=boxWidth,
            patch_artist=True,
            showfliers=False,
        )
        
        # Set different colors for each box within a category
        for box in bx['boxes']:
            box.set_facecolor(colors[i])
        
        handle = Rectangle((0, 0), 1, 1, color=colors[i], label=labels[i])
        handles.append(handle)
   # Set x-axis ticks and labels
    
    ax1.set_xticks([pos + (len(datasets) // 2) * boxWidth for pos in positions])
    ax1.set_xticklabels([f'{x+1}' for x in x_axis])
    ax1.tick_params(axis='x', labelsize=25)  # Set the font size for the x-axis tick labels
    ax1.tick_params(axis='y', labelsize=17) 
    # Combine legends
    plt.legend(handles=handles,fontsize=25)
    plt.title(title,fontsize=30)
    plt.xlabel("Layers",fontsize=25)
    plt.ylabel(y_label,fontsize=25)
    #plt.savefig(title+".pdf")

    plt.show()