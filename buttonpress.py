import numpy as np
from Screencap import screen_grab
import torch
import cv2
import time
import os
import multiprocessing
import torchvision.models as models
import keycapture
from inception_resnet_v2 import Inception_ResNetv2
import random

from pyKey import pressKey, releaseKey, press, sendSequence, showKeys

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

t_time = 0.25


def straight():
    pressKey('W')
    releaseKey('A')
    releaseKey('D')
    releaseKey('S')

def left():
    if random.randrange(0,3) == 1:
        pressKey('W')
    else:
        releaseKey('W')
    pressKey('A')
    releaseKey('S')
    releaseKey('D')


def right():
    if random.randrange(0,3) == 1:
        pressKey('W')
    else:
        releaseKey('W')
    pressKey('D')
    releaseKey('A')
    releaseKey('S')
    
def reverse():
    pressKey('S')
    releaseKey('A')
    releaseKey('W')
    releaseKey('D')


def forward_left():
    pressKey('W')
    pressKey('A')
    releaseKey('D')
    releaseKey('S')
    
    
def forward_right():
    pressKey('W')
    pressKey('D')
    releaseKey('A')
    releaseKey('S')

    
def reverse_left():
    pressKey('S')
    pressKey('A')
    releaseKey('W')
    releaseKey('D')

    
def reverse_right():
    pressKey('S')
    pressKey('D')
    releaseKey('W')
    releaseKey('A')

def no_keys():
    if random.randrange(0,3) == 1:
        pressKey('W')
    else:
        releaseKey('W')
    releaseKey('A')
    releaseKey('S')
    releaseKey('D')


model = models.vit_l_32(weights='DEFAULT')
model.heads= torch.nn.Linear(model.hidden_dim, 9)
checkpoint=torch.load('./models/model_checkpoint_balanced_1.pt')
model.load_state_dict(checkpoint['model_dict'])
model.to('cuda')
model.eval()
paused=False
key_listener_process = multiprocessing.Process(target=keycapture.listen_for_keys)
key_listener_process.start()
del checkpoint
last_pick=[]

while(True):
    
    if not paused:
        last_time = time.time()
        screen = screen_grab()
        if len(last_pick)>10:
            last_pick=[]ssssSDA
            print('clear')
        #last_time = time.time()
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (224,224))
        image_float = screen.astype(np.float32)
        image_float = image_float / 255.0
        # run a color convert:
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
     
       
        #cv2.imshow('window',screen)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    pause=True

        screen=torch.from_numpy(image_float).permute(2,1,0).unsqueeze(0).to('cuda').to(torch.float32)

        output=model(screen)
        #print(output[0][:])
        #top=torch.topk(output,2)
        #print(top)
        output=torch.argmax(output)
        
        r= random.random()
        #output=10
        #choice_picked=' '
        
        if output == 0:
            straight()
            choice_picked = 'straight'                
        elif output == 1:
            reverse()
            choice_picked = 'reverse'            
        elif output == 2:
            if r>0.7:    
                left()
                choice_picked = 'left'
            else:
                forward_left()
                choice_picked='forward+left'
        elif output == 3:
            if r>0.7:
                right()
                choice_picked = 'right'
            else:
                forward_right()
                choice_picked='forward+right'
        elif output == 4:
            choice_picked = 'forward+left'
            if 'forward+right' in last_pick:
                straight()
                choise_picked='straight w hack'
            else:
                forward_left()
            
        elif output == 5:
            choice_picked = 'forward+right'
            if 'forward+left' in last_pick:
                straight()
            else:
                forward_right()
            
        elif output == 6:
            reverse_left()
            choice_picked = 'reverse+left'
        elif output == 7:
            reverse_right()
            choice_picked = 'reverse+right'
        elif output == 8:
            no_keys()
            choice_picked = 'nokeys'

        #print(f'fps:  {1/(time.time()-last_time)}')
        print(choice_picked)
        last_pick.append(choice_picked)
        time.sleep(0.1)

        

                
    keys = keycapture.get_current_keys()
    if ('T' in keys) or ('t' in keys):
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(2)
            keys=None
        else:
            print('Pausing!')
            paused = True
            time.sleep(2)
            Keys=None
