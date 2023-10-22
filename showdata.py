import numpy as np
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from random import shuffle

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

count=np.zeros(shape=9)

for i in range (3):
    train_data=np.load(f'./Train/training_data{i+1}.npy',allow_pickle=True)
    #train_data=np.load(f'./Train/data{i+1}.npy',allow_pickle=True)
    #print(len(train_data))
    count=np.zeros(shape=9)
    for X in train_data:
        X=np.array(X['output'])
        
        count=count+X
        
    #print(f'{i+1}   {count}')
        
    print(f'{count}>>>{i+1}')


