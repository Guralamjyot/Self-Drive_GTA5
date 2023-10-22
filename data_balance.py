import numpy as np
import random
import torch

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]
W,A,S,D,WA,WD,SA,SD,NK=0,0,0,0,0,0,0,0,0
wmaxed,amaxed,smaxed,dmaxed,wamaxed,wdmaxed,samaxed,sdmaxed,nkmaxed=True,True,True,True,True,True,True,True,True

# Define the directory where your split data files are stored.
data_directory = './Train/'

# List to store the data from all split files.
balanced_data = [] 
z=1
# Load data from all split files.
starting_value = 1
while True:
    file_name = f'training_data{starting_value}.npy'
    try:
        data = np.load(data_directory + file_name, allow_pickle=True)
        data=np.array(data)
        starting_value += 1
 
        max_limit=3721
        for item in data:
                
                output=np.argmax(item['output'])               
                if output == 0 and wmaxed:
                    W+=1
                    if W>max_limit:
                        wmaxed=False
                    balanced_data.append(item)                
                elif output == 1 and smaxed:
                    S+=1
                    if S>max_limit:
                        smaxed=False
                    balanced_data.append(item)            
                elif output == 2 and amaxed:
                    A+=1
                    if A>max_limit:
                        amaxed=False
                    balanced_data.append(item)
                elif output == 3 and dmaxed:
                    D+=1
                    if D>max_limit:
                        dmaxed=False
                    balanced_data.append(item)
                elif output == 4 and wamaxed:
                    WA+=1
                    if WA>max_limit:
                        wamaxed=False
                    balanced_data.append(item)
                elif output == 5 and wdmaxed:
                    WD+=1
                    if WD>max_limit:
                        wdmaxed=False
                    balanced_data.append(item)
                elif output == 6 and samaxed:
                    SA+=1
                    if SA>max_limit:
                        samaxed=False
                    balanced_data.append(item)
                elif output == 7 and sdmaxed:
                    SD+=1
                    if SD>max_limit:
                        sdmaxed=False
                    balanced_data.append(item)
                elif output == 8 and nkmaxed:
                    NK+=1
                    if NK>max_limit:
                        nkmaxed=False
                    balanced_data.append(item)
                if len(balanced_data)>=1024:
                    balanced_data=np.array(balanced_data)
                    np.save(f'./Train/balanced_data{z}.npy',balanced_data)
                    z+=1
                    balanced_data=[]
    except FileNotFoundError:
        break