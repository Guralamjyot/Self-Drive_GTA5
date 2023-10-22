import numpy as np
import torch 
from tqdm import tqdm
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from torch.utils.tensorboard import SummaryWriter
def xavier_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

FILE_I_END = 3
LR = 3e-5
EPOCHS = 45

checkpoint=torch.load('./models/model_checkpoint_balanced_1.pt')
model = models.vit_l_32(weights='DEFAULT')


#print(model.heads)
model.heads= torch.nn.Linear(model.hidden_dim, 9)
weights=torch.tensor([1.16,3,1.2,1.23,1,1.15,5,3,1.5],device='cuda')
lossfunction=torch.nn.CrossEntropyLoss(weights)
#model.apply(xavier_init)
#model.load_state_dict(torch.load('./models/model1.pt'))

model.load_state_dict(checkpoint['model_dict'])
global_step=0
#global_step=checkpoint['global_step']
#print(global_step)

model.to('cuda')
optimizer=torch.optim.AdamW(model.parameters(),LR)
optimizer.load_state_dict(checkpoint['optimizer_dict'])

 




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


writer = SummaryWriter()
count=np.zeros(shape=9)
epsilon = 1e-9
step=7748*global_step
for e in range(global_step,EPOCHS):
    del checkpoint
    #model.cuda()
    print(f'epoch: {e+1}')
    data_order = [i for i in range(1,FILE_I_END+1)]
    for i in data_order:        
            file_name = f'./Train/training_data{i}.npy'
            train = np.load(file_name,allow_pickle=True)
            #correct_matrix = np.zeros((9,9))
            #incorrect_matrix = np.zeros((9,9))
            matrix = np.zeros((9,9))
            l=len(train)
            X = np.array([i['screen'] for i in train])
            X=torch.from_numpy(X).permute(0,3,2,1)
            Y = np.array([i['output'] for i in train])  
            Y=torch.from_numpy(Y)
            #X_test = np.array([i['screen'] for i in test])
            #Y_test = np.array([i['output'] for i in test])
            z=0
            del train
            
            optimizer.zero_grad()
            for x,y in tqdm(zip(X,Y)):
                x_cuda=x.to(torch.float32).to('cuda')
                y_cuda=y.to(torch.float32).to('cuda')
                step+=1
                z=z+1
                output=model(x_cuda.unsqueeze(0))

                """ if true_label==predicted_label:
                    correct_matrix[true_label,predicted_label]+=1
                else:
                    incorrect_matrix[true_label,predicted_label]+=1 """
                
                loss=lossfunction(output,y_cuda.unsqueeze(0))  
                loss.backward()
                writer.add_scalar('Loss', loss.item(), step)
                _,output=torch.max(output,1)
                _,true_label=torch.max(y_cuda,0)
                output=output.item()
                matrix[true_label,output]+=1
                
                
                if z%128==0: 
                    optimizer.step()
                    optimizer.zero_grad()
                    #print(torch.argmax(output).item())
                    #print(loss.item())
                if z%512==0:    
                    #normalizedC = correct_matrix / (correct_matrix.sum(axis=1, keepdims=True)+ epsilon)     
                    #normalizedI = incorrect_matrix / (incorrect_matrix.sum(axis=1, keepdims=True)+ epsilon)
                    #heatmapC = (normalizedC * 255).astype(np.uint8)
                    #heatmapI = (normalizedI * 255).astype(np.uint8)             
                    #writer.add_image(f'correct,D:{i}',correct_matrix,global_step=e+1,dataformats='HW')
                    #writer.add_image(f'incorrect,D:{i}',incorrect_matrix,global_step=e+1,dataformats='HW')
                    matrix_img=plt.matshow(matrix,cmap=plt.cm.Blues)
                    mat_file=f'Pngs/mat_newdataV1{step/512}.png'
                    plt.colorbar()
                    plt.savefig(mat_file)
                    plt.close()
                    writer.add_image(f'ViT-newdata_v1',imageio.v2.imread(mat_file),global_step=step/512,dataformats='HWC')

    global_step+=FILE_I_END
    #model.cpu()
    checkpoint={
        'epoch':e,
        'model_dict':model.state_dict(),
        'optimizer_dict':optimizer.state_dict(),
        'global_step':global_step,
    }
    ver=e%2
    torch.save(checkpoint, f'./models/model_checkpoint_balanced_{ver+1}.pt')
    
writer.close()

            
            
                    

            
    


