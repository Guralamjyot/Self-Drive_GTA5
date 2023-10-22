import torch
import numpy as np
count=np.zeros(shape=9)
mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')

for count,i in enumerate(range(27)):
    
        file_name = f'./Train/training_data{i+1}.npy'
        train_data = np.load(file_name,allow_pickle=True)
    


        X = np.array([i['screen'] for i in train_data])
        X=torch.from_numpy(X).to('cuda').permute(0,3,2,1).to(torch.float32)
        print(X[1].mean())
        X = (X - mean[:, None, None]) / std[:, None, None]
        print (X[1].mean())
        break

 