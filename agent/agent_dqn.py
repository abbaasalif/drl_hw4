import torch
import torch.nn.functional as F
import numpy as np
import random
import collections
from agent.agent import Agent
from agent.model import DQN

np.random.seed(1009)
torch.cuda.manual_seed(1009)
torch.manual_seed(1009)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)


def prepro(o):
    o = np.transpose(o, (2,0,1))
    o = np.expand_dims(o, axis=0)
    return o


class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        
        if args.test_dqn:
            print('loading trained model')
            model = torch.load(args.model_name+".ckpt")
            self.current_net = DQN(84, 84)
            self.current_net.load_state_dict(model['current_net'].state_dict())
            self.hyper_param = args.__dict__
            self.current_net = self.current_net.to(device)
        elif args.train_dqn:
            self.current_net = DQN(84, 84)
            self.target_net = DQN(84, 84)
            self.update_target_net()
            self.step_count = 0
            self.epsilon = 1.0
            self.replay_buffer_len = 10000
            self.replay_buffer = collections.deque([], maxlen=self.replay_buffer_len)
            self.optimizer = ['Adam', 'RMSprop', 'SGD']

            self.hyper_param = args.__dict__
            self.training_curve = []

            if self.hyper_param['optim'] in self.optimizer:
                if self.hyper_param['optim'] == 'Adam':
                    self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr = self.hyper_param['learning_rate'], betas = (0.9, 0.999))
                elif self.hyper_param['optim'] == 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(self.current_net.parameters(), lr = self.hyper_param['learning_rate'], alpha = 0.9)
                elif self.hyper_param['optim'] == 'SGD':
                    self.optimizer = torch.optim.SGD(self.current_net.parameters(), lr = self.hyper_param['learning_rate'])
            else:
                print("Unknown Optimizer!")
                exit()
        print(self.current_net)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        self.current_net = self.current_net.to(device)
        self.target_net = self.target_net.to(device)
        
        batch_size = self.hyper_param['batch_size']

        #############################################################
        # YOUR CODE HERE                                            #
        # At the end of train, you need to save your model for test #
        #############################################################
        for episode in range(self.hyper_param['episode']):
            observation = self.env.reset()
            observation = prepro(observation)
            done = False
            total_reward = 0
            while not done:
                action = self.make_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                next_observation = prepro(next_observation)
                total_reward += reward
                self.replay_buffer.append((observation, action, reward, next_observation, done))
                observation = next_observation
                self.step_count += 1
                if self.step_count % 1000 == 0:
                    self.update_target_net()
                # if self.step_count % 10000 == 0:
                #     self.update_epsilon()
                if len(self.replay_buffer) > batch_size:
                    batch = random.sample(self.replay_buffer, batch_size)
                    state_shape = batch[0][0].shape
                    # print(state_shape)
                    batch_observation = torch.Tensor([x[0] for x in batch]).squeeze(dim=1).to(device)
                    batch_action = torch.Tensor([x[1] for x in batch]).to(device)
                    batch_reward = torch.Tensor([x[2] for x in batch]).to(device)
                    batch_next_observation = torch.Tensor([x[3] for x in batch]).squeeze(dim=1).to(device)
                    batch_done = torch.Tensor([x[4] for x in batch]).to(device)
                    q_value = self.current_net(batch_observation)
                    q_value = torch.gather(q_value, 1, batch_action.long().unsqueeze(1)).squeeze(1)
                    next_q_value = self.target_net(batch_next_observation)
                    next_q_value = torch.max(next_q_value, dim=1)[0]
                    expected_q_value = batch_reward + (1 - batch_done) * self.hyper_param['gamma'] * next_q_value
                    loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
                    #update current network every 10 steps
                    if self.step_count % 10 == 0:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()
                    self.update_epsilon()
            print("Episode: {}, Reward: {}, Epsilon: {}".format(episode, total_reward, self.epsilon))
            self.training_curve.append(total_reward)
            if episode % 100 == 0:
                model = {'current_net': self.current_net, 'target_net': self.target_net}
                torch.save(model, "dqn_model_{}.ckpt".format(episode))
        model = {'current_net': self.current_net, 'target_net': self.target_net}
        torch.save(model, "dqn_model.ckpt")
        np.save("dqn_training_curve.npy", self.training_curve)


    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """

        # We only need to use 3 actions. 
        # The index and action is defined below
        #       action 3: left, action 2: right, action 1: stay
        # Note: the returned action is incremented by 1 

        if not test:
            q_value = self.current_net(torch.Tensor(observation).to(device))

            if np.random.rand() < self.epsilon:
                action = np.random.randint(3)
                return action+1
            else:
                action = torch.argmax(q_value)
                return action.item()+1
        else:
            observation = prepro(observation)
            q_value = self.current_net(torch.Tensor(observation).to(device))
            return torch.argmax(q_value).item()+1
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
        
    def update_epsilon(self):
        if self.epsilon >= 0.025:
            self.epsilon -= 0.000001
