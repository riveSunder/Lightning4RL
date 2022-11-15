import argparse
import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import lightning as pl

# rl imports
import gym
import procgen

#l4rl imports
from l4rl.utils.enjoy import enjoy


def get_kwarg(key, default, **kwargs):

    if key in kwargs.keys():
        return kwargs[key]
    else:
        return default

class DQN(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        # learning parameters
        self.lr = get_kwarg("lr", 3e-6, **kwargs)
        # use double dqn?
        self.use_double = True

        # parameters for epsilon greedy exploration
        self.epsilon = get_kwarg("epsilon", 0.9, **kwargs) 
        self.epsilon_decay = get_kwarg("epsilon_decay", 0.9, **kwargs) 
        self.min_epsilon = 0.05
        # discount factor for future time steps
        self.gamma = get_kwarg("gamma", 0.99, **kwargs)

        # architectural parameters
        # control policy is a simple MLP with one hidden layer
        self.hidden = get_kwarg("hidden", [128], **kwargs)
        if type(self.hidden) != list:
            self.hidden = [self.hidden]

        # size of observation vector
        self.input_dim = get_kwarg("input_dim", 4, **kwargs)
        # size of action vector
        self.action = get_kwarg("action", 15, **kwargs)

        # model architecture
        # for image based environments, use a convolutional feature extractor
        if len(self.input_dim) == 3:
            self.build_conv_policy()

        else:
            self.build_mlp_policy()

        # q_target is periodically updated by the current q_model
        self.update_q_target()

        # keep track of epochs
        self.epoch = 0
        self.update_every = 50

    def build_conv_policy(self):

        channels = self.input_dim[2]
        pixels = self.input_dim[0] * self.input_dim[1]

        self.q_model = nn.Sequential(\
                nn.Conv2d(channels, channels*3, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.Conv2d(channels*3, channels*3, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.Conv2d(channels*3, channels, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.ReLU(),
                nn.Flatten())

        self.q_target = nn.Sequential(\
                nn.Conv2d(channels, channels*3, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.Conv2d(channels*3, channels*3, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.Conv2d(channels*3, channels, \
                    kernel_size=3,\
                    stride=2,\
                    padding=1,\
                    padding_mode="circular"),\
                nn.ReLU(),
                nn.Flatten())

        self.hidden =  channels * (pixels // 8**2)

        if type(self.hidden) != list:
            self.hidden = [self.hidden]

        self.hidden += [self.action]

        for index, hidden_nodes in enumerate(self.hidden[1:]):

            if index == len(self.hidden)-1:
                break

            self.q_model.add_module(f"layer_{index}",\
                    nn.Linear(self.hidden[index], hidden_nodes))

            self.q_target.add_module(f"layer_{index}",\
                    nn.Linear(self.hidden[index], hidden_nodes))

    def build_mlp_policy(self):

        self.q_model = nn.Sequential(nn.Linear(self.input_dim[0], self.hidden[0]), nn.ReLU())
        self.q_target = nn.Sequential(nn.Linear(self.input_dim[0], self.hidden[0]), nn.ReLU())

        if type(self.hidden) != list:
            self.hidden = [self.hidden]

        self.hidden += [self.action]

        for index, hidden_nodes in enumerate(self.hidden[1:]):

            if index == len(self.hidden)-1:
                break

            self.q_model.add_module(f"layer_{index}",\
                    nn.Linear(self.hidden[index], hidden_nodes))

            self.q_target.add_module(f"layer_{index}",\
                    nn.Linear(self.hidden[index], hidden_nodes))
        
    def update_q_target(self):

        self.q_target.load_state_dict(copy.deepcopy(self.q_model.state_dict()))

        for parameter in self.q_target.parameters():
            parameter.requires_grad = False

        self.epsilon = max([self.min_epsilon, \
                self.epsilon*self.epsilon_decay])

    def forward(self, obs):
        
        raw_action = self.q_model(obs)

        return raw_action 

    def get_action(self, obs):

        raw_action = self.forward(obs)

        action = torch.argmax(raw_action,-1).numpy()

        return action

    def compute_q_loss(self, batch):

        l_obs, l_act, l_rew, l_next_obs, l_done = batch

        with torch.no_grad():

            q_target_value = self.q_target.forward(l_next_obs)

            if self.use_double:
                q_model_output = self.q_model.forward(l_next_obs)
                qt_max = torch.gather(q_target_value, -1,\
                        torch.argmax(q_model_output, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(q_target_value, -1, \
                        torch.argmax(qt, dim=-1).unsqueeze(-1))

            # add to reward the q_target value for the next step (x discount gamma)
            # unless done is True 
            next_step_reward = l_rew + ((1-l_done) * self.gamma * qt_max)

        l_act = l_act.long()
        # action values predicted for observations
        q_av = self.q_model.forward(l_obs)
        # action values for the actions actually taken
        q_act = torch.gather(q_av, -1, l_act)

        loss =  F.mse_loss(next_step_reward, q_act)

        return loss 

    def training_step(self, batch, batch_idx):
    
        self.zero_grad() 
        
        loss = self.compute_q_loss(batch)

        self.log("train_loss", loss)

        self.epoch += 1

        if self.epoch % self.update_every == 0:
            self.update_q_target()
            
        return loss

    def get_rollout(self, env, max_steps=10000, my_device="cuda"):

        # get trajectories for training

        # arrays to hold different aspects of trajectory
        observations = torch.Tensor().to(my_device)
        rewards = torch.Tensor().to(my_device)
        actions = torch.Tensor().to(my_device)
        next_observations = torch.Tensor().to(my_device)
        dones = torch.Tensor().to(my_device)


        done = True

        self.to(my_device)

        # turn grad off, ensure we don't waste memory
        with torch.no_grad():

            for step in range(max_steps):

                if done:
                    # reset environment after episode
                    obs = env.reset()
                    if len(self.input_dim) == 3:
                        #height, width, channels
                        h, w, c = self.input_dim
                        obs = torch.Tensor(obs.reshape(1, c, h, w)).to(my_device)
                    else:
                        obs = torch.Tensor(obs.reshape(1, -1)).to(my_device)


                    done = False

                if torch.rand(1) < self.epsilon:
                    action = env.action_space.sample()
                else:
                    try:
                        q_values = self.q_model(obs)
                    except:
                        import pdb; pdb.set_trace()
                    action = torch.argmax(q_values, dim=-1)
                    action = action.detach().cpu().numpy()[0]

                previous_obs = obs
                obs, reward, done, info = env.step(action)

                if len(self.input_dim) == 3:
                    obs = torch.Tensor(obs.reshape(1, c, h, w)).to(my_device)

                else:
                    obs = torch.Tensor(obs.reshape(1, -1)).to(my_device)


                # concatenate current step into buffers

                observations = torch.cat([observations, previous_obs], dim=0)
                next_observations = torch.cat([next_observations, obs], dim=0)

                reward = torch.Tensor(np.array(1.* reward)).to(my_device)
                action = torch.Tensor(np.array(1.* action)).to(my_device)
                done = torch.Tensor(np.array(1.* done)).to(my_device)

                rewards = torch.cat([rewards, \
                        reward.reshape(1,1)], dim=0)
                actions = torch.cat([actions, \
                        action.reshape(1,1)], dim=0)
                dones = torch.cat([dones, \
                        done.reshape(1,1)], dim=0)

    

        return observations.cpu(), actions.cpu(), rewards.cpu(),\
                next_observations.cpu(), dones.cpu()

        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), \
                lr=self.lr)

        return optimizer
       
class Trajectory():

    def __init__(self, rollout):
        self.rollout = rollout

    def __len__(self):

        return len(self.rollout[0])

    def __getitem__(self, indices):

        obs, act, rew, next_obs, dones = [self.rollout[ii][indices] \
                for ii in range(len(self.rollout))]

        return (obs, act, rew, next_obs, dones)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, \
            default=1000,\
            help="number of steps to include in each training step batch")
    parser.add_argument("-e", "--epochs", type=int, \
            default=10,\
            help="epochs to train on each rollout")
    parser.add_argument("-l", "--learning_rate", type=float , \
            default=3e-6,\
            help="learning rate")
    parser.add_argument("-m", "--max_steps", type=int, \
            default=10000,\
            help="total number of steps per rollout")
    parser.add_argument("-r", "--rollouts", type=int, \
            default=100,\
            help="number of rollouts to roll out")
    parser.add_argument("-s", "--seed", type=int,\
            default=42,\
            help="random seed")
    parser.add_argument("-t", "--tag", type=str,\
            default="default_exp",\
            help="tag to identify experiment run")
    parser.add_argument("-z", "--ez_mode", type=int, \
            default=0,\
            help="train in easy mode (CartPole env) unless ez_mode is 0")

    args = parser.parse_args()

    batch_size = args.batch_size
    ez = args.ez_mode
    exp_tag = f"{args.tag}_{int(time.time())}"
    rollouts = args.rollouts
    max_steps = args.max_steps
    max_epochs = args.epochs
    lr = args.learning_rate
    my_seed = args.seed

    np.random.seed(my_seed)
    torch.manual_seed(my_seed)

    # bigfish, starpilot, fruitbot, climber
    if not ez:
        env_name = "procgen:procgen-chaser-v0"
        env = gym.make(env_name, distribution_mode="easy", start_level=0, num_levels=1)
    else:
        env_name = "CartPole-v1"
        env = gym.make(env_name)

    input_dim = env.observation_space.shape
    action_dim = env.action_space.n

    dqn = DQN(input_dim=input_dim, action=action_dim, lr=lr)

    # before training (basically a random agent)
    enjoy(dqn, env, render=False, total_steps=5000)

    try:
        for gen in range(rollouts):
            if ez:
                env = gym.make(env_name) #, distribution_mode="easy", start_level=0, num_levels=1)
            else:
                env = gym.make(env_name, distribution_mode="easy", start_level=0, num_levels=1)
            t0  = time.time()
            rollout = dqn.get_rollout(env, max_steps=max_steps)
            t1 = time.time()

            print(f"rollout time {t1-t0:.3f}")
            print(f"gen {gen} mean reward: {np.mean(rollout[2].cpu().numpy())}"
                    f", mean steps/epd: {1/np.mean(rollout[4].cpu().numpy())}")

            trajectory = Trajectory(rollout)

            idx = np.random.choice(len(trajectory), batch_size, replace=False)

            temp = trajectory[idx]

            dataloader = DataLoader(trajectory, batch_size=batch_size, num_workers=16)

            if torch.cuda.is_available():
                trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs)
            else:
                trainer = pl.Trainer(max_epochs=5)

            trainer.fit(model=dqn, train_dataloaders=dataloader)
            enjoy(dqn, env, render=False, total_steps=250)

            torch.save(dqn.state_dict(), f"models/{exp_tag}.pt")

    except KeyboardInterrupt:
        import pdb; pdb.set_trace()

    if ez:
        env = gym.make(env_name) #, distribution_mode="easy", start_level=0, num_levels=1)
    else:
        env = gym.make(env_name, render_mode="human", distribution_mode="easy", start_level=0, num_levels=1)

    enjoy(dqn, env, render=True)
