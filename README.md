# Introduction 
It started as a hands-on project with some parctical goal in mind, along with the intent of learning I/O functionality of python on linux. As it progressed, it turned form a CNN based classification problem a ViT one, with the intuition that a ViT would be able to "Attend" to different objects according to their position on the screen,
leading to a better judgement by the AI and overall a better performing model, along with the better performance of transformers as compared to CNNs it was an obvious choise(my training time was reduced to half as compared to using a Resnet_101). Also, trying to get the model to converge without any transfer learning was an almost impossible task, with the model refusing to go below a satisfactory loss
or accuracy.


# Usage
The enitre codebase is written to work on linux, you can switch up screencap and keycapture code to get it working on windows aswell, should be a fairly simple task which i might end up doing in the future.
You need approximately 16gigs of RAM and 8gigs of Vram for the ViT_L_32, and more for the versions with smaller patches.  

1- You start by collecting the required data for the model. running collectdata.py would do that, make sure the game is launched in windowed mode, with the resolution of 800x600.
The script starts collecting the data when you press 'T' which acts as a play/pause switch to make life a little easier. The script is configured in a way to only a new datapoint after a delay of 0.015 essentially making it record in ~60fps.
To make the data a little more balanced, script is also made to only accept a certain percent of certain inputs which are over-represented in my playstyle; this increases the quality of data and makes the model learn more useful features even with the same number of data-records.
The data is stored in the directory inside 'Train' with each datafile containing 4096 samples of screen-captures and the buttons pressed at that moment.
It is recommended to always drive to a location marked by a waypoint, it gives the AI a sense of 'purpose' instead of driving around for no reason, and watching it follow the route is honeslty heartwarming. :P 
It is also recommened to drive a slow vehicle(a van would do fine) with hood-camera instead of in-car camera of the first person mode.

2- You can no chose to get a sense of the data collected by running showdata.py, it prints out the distribution of 9 target values corrosponding to each button input(W,S,A,D,WA,WD,SA,SD and no_key)

3- If you still have some severe imbalances which cannot be overcome by using a weighted loss function, you can try running the data_balance.py script after configuring the parameters.
Thought if would be a good idea to run this regardless, to shuffle the data around and get well distributed batches.

4- Run the train_Vit.py script to get the model learning, you can adjust the values of weight tensor according to the distribution of your data. It is configured to save a new checkpoint after every epoch.
Also generates a confusion matrix in '/Pngs' after every 512 data samples. The script assumes you have a cuda device available, anything with 8gigs of Vram would do just fine since we're using gradient accumilation to get an effective batch size of 128.

5- After training, once the game is loaded, run Buttonpress.py and switch focus to game window and let the AI take over.


# Future improvements
1- Visualize the attention layer in real-time to see what the agent is "looking" at.

2- Create a reinforcement learning environment for the agent. 

3- Get more data, different ingame time, different terrains and vehicles with varying camera-angles 
