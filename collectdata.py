import numpy as np
from Screencap import screen_grab
import cv2
import multiprocessing
import time
import keycapture
import random
import os

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

starting_value = 1

while True:
    file_name = f'Train/training_data{starting_value}.npy'

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'w' in keys and 'a' in keys:
        output = wa
    elif 'w' in keys and 'd' in keys:
        output = wd
    elif 's' in keys and 'a' in keys:
        output = sa
    elif 's' in keys and 'd' in keys:
        output = sd
    elif 'w' in keys:
        output = w
    elif 's' in keys:
        output = s
    elif 'a' in keys:
        output = a
    elif 'd' in keys:
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_value):
    key_listener_process = multiprocessing.Process(target=keycapture.listen_for_keys)
    key_listener_process.start()
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)

    #last_time = time.time()
    paused = True
    print('STARTING!!!')
    while(True):
        if not paused:
            rand=random.random()

            screen = screen_grab()
            #last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (224,224))
            image_float = screen.astype(np.float32)
            screen = image_float / 255.0

            # run a color convert:
            #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = keycapture.get_current_keys()
            output = keys_to_output(keys)

            if (output== w) and (rand<0.05):
                training_data.append([screen,output])
            
            elif (output== wa) and (rand<0.5):
                if rand<0.1:
                    output=a
                training_data.append([screen,output])
            
            elif (output== wd) and (rand<0.5):
                if rand<0.1:
                    output=d
                training_data.append([screen,output])                    

            elif (output== nk) and (rand<0.4):
                training_data.append([screen,output])

            elif output!=w and output!=wa and output!=wd and output!=nk:
                training_data.append([screen,output])
                #time.sleep(0.02)

            time.sleep(0.015)

            cv2.imshow('window',screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if (len(training_data) % 4096 == 0) and (len(training_data)!=0):
                    data = [{'screen': screen, 'output': output} for screen, output in training_data]
                    training_data = np.array(data)
                    np.save(f'./Train/training_data{starting_value}.npy',training_data)
                    print(f'SAVED: {starting_value}')

                    training_data = []
                    starting_value += 1
                    file_name = f'training_data.npy{starting_value}'

                    
        keys = keycapture.get_current_keys()
        if 't' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        keys=None

main(file_name, starting_value)