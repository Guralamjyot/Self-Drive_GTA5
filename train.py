import numpy as np
import torch 
from tqdm import tqdm
import torchvision.models as models
from tqdm import tqdm
import torchvision.transforms as transforms
from inception_resnet_v2 import Inception_ResNetv2

def xavier_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


model = Inception_ResNetv2()
#model.apply(xavier_init)

model.load_state_dict(torch.load('./models/model6.pt'))

model.to('cuda')



FILE_I_END = 62
LR = 1e-3
EPOCHS = 12



w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]



# iterates through the training files
weights=torch.tensor([0.2,4,2,2,0.6,0.6,5,2,1],device='cuda')
lossfunction=torch.nn.CrossEntropyLoss(weights)
optimizer=torch.optim.AdamW(model.parameters())

for e in range(EPOCHS):
    print(f'epoch: {e+1}')
    data_order = [i for i in range(1,FILE_I_END+1)]
    for i in data_order:        
            file_name = f'./Train/training_data{i}.npy'
            train = np.load(file_name,allow_pickle=True)

            X = np.array([i['screen'] for i in train])
            X=torch.from_numpy(X).to('cuda').permute(0,3,1,2).to(torch.float32)
            Y = np.array([i['output'] for i in train])  
            Y=torch.from_numpy(Y).to('cuda').to(torch.float32)
            #X_test = np.array([i['screen'] for i in test])
            #Y_test = np.array([i['output'] for i in test])
            z=0
            save=0
            optimizer.zero_grad()
            for x,y in tqdm(zip(X,Y)):
                output=model(x.unsqueeze(0))
                #print(output)
                loss=lossfunction(output,y.unsqueeze(0))
                loss.backward()
                z=z+1
                if z%128==0: 
                    optimizer.step()
                    optimizer.zero_grad()
                    #print(torch.argmax(output).item())
                    #print(loss.item())
                if z==1000:
                    optimizer.step()
                    optimizer.zero_grad()                     
            
            if i%2==0:
                torch.save(model.state_dict(), f'./models/model{e+1}.pt')

            
            
                    

            
    


