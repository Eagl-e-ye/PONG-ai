import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from game_env import PongGame
from experience_replay import ReplayMemory
from n_network import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import threading

import os


DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']              # size of 1st hidden layer
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.train_left = hyperparameters['train_left']
        self.train_right = hyperparameters['train_right']


        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE_L = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_L.pt')
        self.MODEL_FILE_R = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_R.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def save_model(self, log_message, dqn, epsilon, step_count, best_reward,optimizer, model_file):
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n' )
            torch.save({
                'model_state_dict': dqn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'step_count': step_count,
                'best_reward': best_reward
            }, model_file)
        

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
                
        env = PongGame(render_mode= 'human')
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episodes = []

        left_dqn =DQN(num_states, num_actions, self.fc1_nodes).to(device)
        right_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        
        self.optimizer_L = torch.optim.Adam(left_dqn.parameters(), lr= self.learning_rate_a)
        self.optimizer_R = torch.optim.Adam(right_dqn.parameters(), lr= self.learning_rate_a)
        
        step_count = 0
        
        if os.path.exists(self.MODEL_FILE_L) and is_training:
            checkpoint_L = torch.load(self.MODEL_FILE_L)
            left_dqn.load_state_dict(checkpoint_L['model_state_dict'])
            self.optimizer_L.load_state_dict(checkpoint_L['optimizer_state_dict'])
            epsilon = checkpoint_L['epsilon']
            step_count = checkpoint_L['step_count']
            best_reward_L = checkpoint_L['best_reward']
            print("loaded left")

        if os.path.exists(self.MODEL_FILE_R) and is_training:
            checkpoint_R = torch.load(self.MODEL_FILE_R)
            right_dqn.load_state_dict(checkpoint_R['model_state_dict'])
            self.optimizer_R.load_state_dict(checkpoint_R['optimizer_state_dict'])
            epsilon = checkpoint_R['epsilon']
            step_count = checkpoint_R['step_count']
            best_reward_R = checkpoint_R['best_reward']
            print("loaded right")



        if is_training:
            memory_l = ReplayMemory(self.replay_memory_size)
            memory_r = ReplayMemory(self.replay_memory_size)


            tar_left_dqn =DQN(num_states, num_actions, self.fc1_nodes).to(device)
            tar_left_dqn.load_state_dict(left_dqn.state_dict())
            tar_right_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            tar_right_dqn.load_state_dict(right_dqn.state_dict())

            if not os.path.exists(self.MODEL_FILE_L):
                epsilon = self.epsilon_init
                best_reward_L = -99999999
            if not os.path.exists(self.MODEL_FILE_R):
                epsilon = self.epsilon_init
                best_reward_R = -99999999

            epsilon_history = []
            

        else:
            left_dqn.load_state_dict(torch.load(self.MODEL_FILE_L)['model_state_dict'])
            right_dqn.load_state_dict(torch.load(self.MODEL_FILE_R)['model_state_dict'])

            left_dqn.eval()
            right_dqn.eval()

        for episode in itertools.count():
            state , _ =env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)


            terminated = False
            episode_reward_l = 0.0
            episode_reward_r = 0.0

            while not terminated and (episode_reward_l < self.stop_on_reward or episode_reward_r < self.stop_on_reward):
                if is_training and random.random()< epsilon:
                    left_action = env.action_space.sample()
                    left_action = torch.tensor(left_action, dtype=torch.int64, device=device)
                    right_action = env.action_space.sample()
                    right_action = torch.tensor(right_action, dtype=torch.int64, device=device)

                else:
                    with torch.no_grad():
                        left_action = left_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        right_action = right_dqn(state.unsqueeze(dim=0)).squeeze().argmax()



                        #only for manual control
                        # keys = pygame.key.get_pressed()
                        # if keys[pygame.K_w] or keys[pygame.K_UP]:
                        #     right_action = torch.tensor([0], device=device)
                        # elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                        #     right_action = torch.tensor([1], device=device)
                        # else:
                        #     # Default action (e.g., stay still or some neutral action)
                        #     right_action = torch.tensor([2], device=device)

                        

                new_state, reward, terminated, _ , info= env.step(left_action.item(), right_action.item())
                reward_l=reward[0]
                reward_r=reward[1]

                episode_reward_l += reward_l
                episode_reward_r += reward_r

                new_state = torch.tensor(new_state, dtype=torch.float, device= device)
                reward_l = torch.tensor(reward_l, dtype=torch.float, device=device)
                reward_r = torch.tensor(reward_r, dtype=torch.float, device=device)

                if is_training:
                    if self.train_left:
                        memory_l.append((state, left_action, new_state, reward_l, terminated))
                    if self.train_right:
                        memory_r.append((state, right_action, new_state, reward_r, terminated))

                    step_count+=1

                state= new_state


            rewards_per_episodes.append((episode_reward_l, episode_reward_r))

            if is_training:
                if self.train_left and episode_reward_l > best_reward_L:
                    log_message_l = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward left {episode_reward_l:0.1f} ({(episode_reward_l-best_reward_L)/best_reward_L*100:+.1f}%) at episode {episode}, saving model..." if best_reward_L !=0 else f"{datetime.now().strftime(DATE_FORMAT)}: New best reward left {episode_reward_l:0.1f} (0%) at episode {episode}, saving model..."
                    print(log_message_l)
                    best_reward_L = episode_reward_l
                    #############functions################
                    save_model_l= threading.Thread(target=self.save_model, args=(log_message_l, left_dqn, epsilon, step_count, best_reward_L, self.optimizer_L, self.MODEL_FILE_L))
                    save_model_l.start()

                if self.train_right and episode_reward_r > best_reward_R:
                    log_message_r = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward right {episode_reward_r:0.1f} ({(episode_reward_r-best_reward_R)/best_reward_R*100:+.1f}%) at episode {episode}, saving model..." if best_reward_R !=0 else f"{datetime.now().strftime(DATE_FORMAT)}: New best reward right {episode_reward_r:0.1f} (0%) at episode {episode}, saving model..."
                    print(log_message_r)
                    best_reward_R = episode_reward_r
                    
                    save_model_r= threading.Thread(target=self.save_model, args=(log_message_r, right_dqn, epsilon, step_count, best_reward_R, self.optimizer_R, self.MODEL_FILE_R))
                    save_model_r.start()
                    


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episodes, epsilon_history)
                    last_graph_update_time = current_time



                if self.train_left and len(memory_l) >= self.mini_batch_size:
                    mini_batch_l = memory_l.sample(self.mini_batch_size)
                    self.optimize(mini_batch_l, left_dqn, tar_left_dqn, self.optimizer_L)

                if self.train_right and len(memory_r) >= self.mini_batch_size:
                    mini_batch_r = memory_r.sample(self.mini_batch_size)
                    self.optimize(mini_batch_r, right_dqn, tar_right_dqn, self.optimizer_R)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count >= self.network_sync_rate:
                    if self.train_left:
                        tar_left_dqn.load_state_dict(left_dqn.state_dict())
                    if self.train_right:
                        tar_right_dqn.load_state_dict(right_dqn.state_dict())
                    step_count = 0

    

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        # Separate rewards
        rewards_l = [r[0] for r in rewards_per_episode]
        rewards_r = [r[1] for r in rewards_per_episode]

        # Compute moving average
        mean_rewards_l = []
        mean_rewards_r = []

        for x in range(len(rewards_l)):
            mean_rewards_l.append(np.mean(rewards_l[max(0, x-99):(x+1)]))

        for x in range(len(rewards_r)):
            mean_rewards_r.append(np.mean(rewards_r[max(0, x-99):(x+1)]))

        plt.subplot(131) 
        plt.ylabel('Mean Rewards L')
        plt.plot(mean_rewards_l)

        plt.subplot(132) 
        plt.ylabel('Mean Rewards R')
        plt.plot(mean_rewards_r)

        plt.subplot(133) 
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, dqn, target_dqn, optimizer):
        
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_fn(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)