import torch
from torch import nn
import torch.nn.functional as F
from agent.agent import Agent
from environment import Environment
import numpy as np
import skimage.transform

np.random.seed(1009)
torch.cuda.manual_seed(1009)
torch.manual_seed(1009)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def prepro(o, image_size=[105, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    """

    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]   #gray scale
    resized = skimage.transform.resize(y, image_size)[17:-8,:]            #delete score board
    return np.expand_dims(resized.astype(np.float32), axis=2)             #shape (height, wodth) -> (1, height, wodth)


class Agent_PG(Agent):

    def __init__(self, env, args):
        super(Agent_PG, self).__init__(env)
        
        if args.test_pg:
            print('loading trained model')
            self.model = torch.load(args.model_name + ".ckpt")
            self.hyper_param = args.__dict__
            self.last_frame = None

        elif args.train_pg:
            self.optimizer = ['Adam', 'RMSprop', 'SGD']
            self.hyper_param = args.__dict__
            self.argument = args
            self.training_curve = []
            self.last_frame = None
            
            # initial model
            self.model = nn.Sequential(
                nn.Linear(80*80*1, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ).to(device)
                
            # initial optimizer
            if self.hyper_param['optim'] in self.optimizer:
                if self.hyper_param['optim'] == 'Adam':
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_param['learning_rate'], betas=(0.9, 0.999))
                elif self.hyper_param['optim'] == 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = self.hyper_param['learning_rate'], alpha = 0.9)
                elif self.hyper_param['optim'] == 'SGD':
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.hyper_param['learning_rate'])
            else:
                print("Unknown Optimizer!")
                exit()

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        """
        self.last_frame = None

    def train(self):
        gamma = self.hyper_param['gamma']
        batch = self.hyper_param['batch_size']

        #############################################################
        # YOUR CODE HERE                                            #
        # At the end of train, you need to save your model for test #
        #############################################################
        

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        # We only need two actions. The index and action is defined below 
        #    action 2: up, action 3: down
        
        if not test:
            observation = torch.Tensor(observation).view(1, -1).to(device)
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action, output
        elif test:
            if type(self.last_frame) == type(None):
                observation = prepro(observation)
                self.last_frame = observation
            else:
                o = prepro(observation)
                observation = o - self.last_frame
                self.last_frame = o
            observation = torch.Tensor(observation).view(1, -1).to(device)
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action
