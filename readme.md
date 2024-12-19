Please update this file accordingly, e.g., the train and test scripts, and remove the one you don't implement.

TA will run your test script for evluation. If you use TensorFlow, please add pip install command here.

Please use a CUDA GPU machine as the model is trained in GPU there would be issues loading it in CPU version of pytorch.

For the testing script please dont use .ckpt in the script it is added automatically -run as shown in the usage below.

For DQN
```bash
python main.py --train_dqn --model_name dqn1.ckpt -ep 20000 -b 32 -lr 0.00015
```
```bash
python main.py --test_dqn  --model_name dqn1
```
This plays the Atari game and the current model is hyperparameter optimized using Optuna and surpasses most human players. The weights are not included due to size issues. 
