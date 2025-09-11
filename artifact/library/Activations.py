import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from library.utils import discretize,scotts_rule

import math

class Activations : 
  
    def __init__(self,index,label,prediction,activation_set):
        self.index= index
        self.label = label
        self.prediction =prediction
        self.activations_set = activation_set
        self.start_layer = 0
        self.end_layer = 100
        self.is_clipped = False
     
    # Returns a set represnting the activations weight of the activations shape : [[layer1] , [layer2],....,[layerN]] 
    def get_activations_set(self):
        if(self.is_clipped):
            return self.activations_set[self.start_layer:self.end_layer+1]
        return self.activations_set  

    #returns the true label of the activations
    def get_label(self):
        return self.label

    #return total nb of layers
    def get_nb_layers(self):
        return len(self.activations_set)


    #Saves activations a csv in corresponding folder
    def save_csv(self,folder_name):
        if(not self.activations_set):
            raise ValueError('Activations parameters are  undefined, cannot save')
 
        DF = pd.DataFrame(self.activations_set[0],columns=["Layer_0"]) 
        for i,out in enumerate(self.activations_set):
            if (i == 0 ):
                continue
            DF["Layer_"+ str(i)] = pd.DataFrame(self.activations_set[i])
            # save the dataframe as a csv file
            #Fill in layer 1 information (node values) in a column in the CSV 
        filename = str(self.label) + "_"+ str(self.prediction) +"-"+str(self.index)
      
        DF.to_csv(folder_name +"/"+filename +".csv", index=False)

    #if each node has more than 1 weight( cnn's nodes produce an n*n matrix) it is saved in a differetn format
    def save_cnn(self,activations_list,folder_name):
        layer = 0
        if not (os.path.exists(folder_name)):
            os.makedirs(folder_name)
     
        filename = str(self.label) + "_"+ str(self.prediction) +"-"+str(self.index)+'.txt'
        with open(folder_name+'/'+filename, 'a') as out:
            for i in activations_list :
                out.write(f'Layer :{layer}; \n')
                for index,node in enumerate(i) :
                    out.write(f'Node :{index}; \n')
                    charac = np.array(node).flatten()
                    out.write(np.array2string(charac,max_line_width= float('+inf'),separator=','))
                    out.write('\n')
                layer+=1
            out.close()


    #Generates dot file and saves it in default folder
    def as_dot(self,start_layer =1,end_layer=3):
        file1 = open("graph.txt","w")
        L = ["digraph G {\n", "rankdir = LR;\n", "splines=line;\n","ranksep= 1.4;\n","node [fixedsize=true, label=\"\"];\n"]  
        file1.writelines(L)
        L=["subgraph cluster_0 {\n","color=white;\n","node [style=solid, shape=circle];\n"]
        file1.writelines(L)
        for i,activation in enumerate(self.activations_set):
            if( i< start_layer or i>end_layer):
                continue
            for r in range (0, (activation[0]).size):
                file1.write("L"+ str(i)+"N")
                file1.write(str(r))
                file1.write("[label=N")
                file1.write(str(r))
                file1.write("]")
                if (activation[0].item(r))!=0:
                    file1.write("[shape=circle, style=filled, fillcolor=green]")
                file1.write("\nlabel=\"Layer "+str(i)+"\"")    
                file1.write(" ")

        file1.write("\n}")

        index = start_layer
        while(index != end_layer):
            for r in range (0, (self.activations_set[index][0]).size):
            
                for j in range (0, (self.activations_set[index+1][0]).size):
                    file1.write("L"+str(index)+"N")
                    file1.write(str(r))
                    file1.write("->")
                    file1.write("L"+str(index+1)+"N")
                    file1.write(str(j))
                    file1.write("\n")
            print(index)
            index+=1
        
        file1.write("}")
        file1.close() #to change file access modes
        os.system("dot -Tpng -O graph.txt")
        print('Generated graph dot code and saved in graph.txt, can see graph in graph.txt.png')


     # returns actication set as binary : meaning active_node => 1 , inactive_nodes =>0
    def get_binary(self,threshhold):
        aux = []
        for i,s in enumerate(self.activations_set) :
            aux.append([])
            for a in s :
                #define a threshhold for what active and inactive mean
                if a >threshhold :
                    aux[i].append(1)
                else :
                    aux[i].append(0)
        return aux

    def hamming_index (self,reference_activations,threshhold) :
        #given two binary activations set computes how simialr they are
        # ; how any bits need ti be changed to transform one to the other
        binar_actiovation_set = self.get_binary(threshhold)
        binary_reference_set = reference_activations.get_binary(threshhold)
        index = 0
        for i,x in enumerate(binar_actiovation_set):
            if( i <self.start_layer or i>self.end_layer):
                continue
            
            for j,y in enumerate(x) :
                if(y != binary_reference_set[i][j]):
                    index += 1
            
        return index
         
    def print(self):
        print('label : %s prediction : %s index :%s' %(self.label,self.prediction,self.index))
    
    #returns number of active nodes : weight > threshhold
    def compute_nb_active_nodes(self,threshhold):
        nb = 0
        layer_count = 0
        for i,x in enumerate(self.activations_set):
            if(i<self.start_layer or i>self.end_layer):
                continue
            layer_count+=1
            for j,y in enumerate(x):
                if(type(y) == type([])): continue
                if abs(y) > threshhold:
                    nb +=1
        return nb
    #get total number of node (considering clipping)
    def get_nb_nodes(self):
        return (len(self.flatten()))

    # flatten array  => [ ...[layer1] , ...[layer2] , ..[layer3]]
    def flatten(self):
        if(self.is_clipped):
            slice_list = self.activations_set[self.start_layer:self.end_layer+1]
        else :
            slice_list = self.activations_set
        flat_list = [item for sublist in slice_list for item in sublist]
        #print('Flattened List to %s elements' %(len(flat_list)))
        return flat_list

    # set layers to be considered if clipping is needed ( for a per layer analysis)
    def set_layer_range(self,start_layer,end_layer):
        self.start_layer =start_layer
        if(end_layer == float('+inf')):
            self.end_layer = len(self.activations_set)
        self.is_clipped = True
        return self

    #Get if prediction is equal to label
    def get_truth_value(self):
        return self.prediction == self.label

    # get average activations weight
    def get_average_weight(self,nonZero = True):
        res = 0
        for i,x in enumerate(self.activations_set):
            if( i <self.start_layer or i > self.end_layer):
                continue
            if(len(x)==0):continue
            s= [ i for i in x if (type(i) == type([]))]
            if(len(s) !=0): continue
            res+= np.average(np.array(x))
        
        if(self.is_clipped):
            res = res /(self.end_layer -self.start_layer+1)
        else :
            res = res / len(self.activations_set)
        return res 
    
    
    def get_nb_active_nodes (self,threshhold):
        count = 0
        for i,x in enumerate(self.activations_set):
            if(i<self.start_layer or i>self.end_layer):
                continue
            for j in x :
                if(j>threshhold):
                    count+=1
        return count

    
    #get dipersation index of this activations
    def dispersation_index(self):
        if(self.is_clipped):
            a = self.activations_set[self.start_layer:self.end_layer+1]
            a = [item for sublist in a for item in sublist]
        else:
             a = self.flatten()
        r = []
        for i in a : 
            if(type(i) != type([])):
                r.append(i)
        return np.var(r) * np.var(r) /np.mean(r)    
   

    def drop_and_get(self,nb):
        if(self.is_clipped):
            a = self.activations_set[self.start_layer:self.end_layer+1]
            a = [item for sublist in a for item in sublist]
        else :
             a = self.flatten()
        return  list(filter(lambda x: (abs(x) < nb ), a)) 


    # plot activations
    def plot(self,color ="green",label=''):
        axis = np.arange(len(self.flatten()))
        plt.figure(figsize=(10, 4))
        plt.xlabel("Nodes")
        plt.ylabel("Activaiton weight")       
        plt.scatter(axis, self.flatten(), s=7,color =color)
        plt.title(label=label)

        #plot verticacl line to tresprresents layers seperatio
        prev = 0
        for i in  self.activations_set[self.start_layer:self.end_layer]:
            plt.axvline(x = len(i)+prev, color = 'b', label = 'Layer')
            prev = prev+len(i)
        plt.show()


    def transform_layers_to_image(self,activations):
        max_length = 0
        for i in activations : 
            if(len(i)>max_length ):
                max_length = len(i)
        
        if(max_length ==0):
            raise KeyError
        print(f'max_length {max_length}')
        aux = []
        for i in activations:
            length_to_cover = max_length-len(i)
            for j in range(math.floor(length_to_cover/2)):
                i.insert(0,0)
            for j in range(math.floor(length_to_cover/2)):
                i.append(0) 
            aux.append(i)
        return aux


    def draw_as_image(self,activations=None):

        activationself = self.transform_layers_to_image(self.activations_set)
        activationAux = self.transform_layers_to_image( activations)
        print(activationAux[0])
        print(activationself[0])

        f, axarr = plt.subplots(1,2,figsize=(10,7))
        axarr[0].imshow(activationself,interpolation='nearest', aspect='auto')
        axarr[1].imshow(activationAux,interpolation='nearest', aspect='auto')
        plt.show()


        return

    def plot_single(self,activations ):
        activationAux = self.transform_layers_to_image( activations)
        plt.imshow(activationAux,interpolation='nearest', aspect='auto')
        plt.show()

    # unflatten array to root form
    def deflatten(self,flattened):
        aux = []
        index = 0 
        for i in range(len(self.activations_set)):
            lengthA = len(self.activations_set[i])
            arr = flattened[index:index+lengthA].tolist()
            aux.append(arr)
            index+=lengthA   
        return aux
    
    # get shape of activations (layer1.shape,layer2.shape ,..., layerN.shape)
    def get_layers_shape (self):
        layers_shape = []
        for index,i in enumerate(self.activations_set):
            if(index<self.start_layer or index>self.end_layer):
                continue
            layers_shape.append(len(i))
        return layers_shape

    def is_spoiled(self):
        #For CIfar10 some nodes contain empty array [] {BuG debt} to avoid erros we just replace the array with a weight of 0
        array = self.flatten()
        for i in array:
            if(type(i) == type([])):
                return True
        return False

    def compute_entropy(self):

        if (self.is_spoiled()): return 0
        array = self.flatten()
        nb_bins = scotts_rule(array)
        array = discretize(array,nb_bins)

        array = array.tolist()
        size = len(array)
        unique_elements = set(array)
        entropy = 0.0
        for element in unique_elements:
            count = array.count(element)
            probability = count / size
            entropy += probability * math.log2(probability)
        return -entropy
            
