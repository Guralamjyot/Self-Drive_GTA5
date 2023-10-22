import numpy as np
z=0
group=[]
k=1
for i in range(27):        
    file_name = f'./Train/balanced_data{i+1}.npy'
    train = np.load(file_name,allow_pickle=True)
    for item in train:
        group.append(item)
        z+=1
        if z>=4096:
            np.save(f'./Train/data{k}.npy',group)
            group=[]
            z=0
            k+=1

np.save(f'./Train/data{k}.npy',group)
