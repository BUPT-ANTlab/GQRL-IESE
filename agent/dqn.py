from agent.controller import DeepLearningController, ReplayMemory
from agent.modules import IESE, PursuitModule, UPDeT
from agent.utils import argmax, get_param_or_default
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class DQNNet(torch.nn.Module):
    def __init__(self, input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, outputs, max_history_length, params=None):
        super(DQNNet, self).__init__()
        self.fc_net = IESE(input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, max_history_length, params=params)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, outputs)

    def forward(self, single_obs):
        x = self.fc_net(single_obs)
        return F.softmax(self.action_head(x))



class DQNLearner(DeepLearningController):

    def __init__(self, params):
        super(DQNLearner, self).__init__(params)
        self.epsilon = 1.0
        self.epsilon_decay =  0.0001
        self.epsilon_min =  0.01
        self.batch_size = 3
        history_length = self.max_history_length
        input_ego_state_shape = [8,10]
        input_oth_state_shape= [8,10]
        input_eva_state_shape = [8, 10]
        num_actions = self.num_actions
        network_constructor = lambda in_ego_state_shape, in_oth_state_shape,in_eva_state_shape, actions, length: DQNNet(in_ego_state_shape, in_oth_state_shape,in_eva_state_shape,actions, length, params=params)
        self.policy_net = PursuitModule(input_ego_state_shape, input_oth_state_shape,input_eva_state_shape,num_actions, history_length, network_constructor).to(self.device)
        self.target_net = PursuitModule(input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, num_actions, history_length, network_constructor).to(self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.update_target_network()

    def joint_action_probs(self, histories, obs, mixing_net,training_mode=True, agent_ids=None):
        action_probs = []
        used_epsilon = self.epsilon_min
        if training_mode:
            used_epsilon = self.epsilon
        if agent_ids is None:
            agent_ids = self.agent_ids

        for i, agent_id in enumerate(agent_ids):
            # history = [[joint_obs[i]] for joint_obs in histories]
            # history = torch.tensor(history, device=self.device, dtype=torch.float32)
            if random.random()< used_epsilon:
                Q_values=F.softmax(torch.tensor(np.random.rand(self.num_actions,), device=self.device, dtype=torch.float32)).detach().cpu().numpy()
            else:
                history = [histories[i]]
                Q_values = self.policy_net(history).detach().cpu().numpy()[0]

            action_probs.append(Q_values)
        if mixing_net is not None:
            action_probs_fin= mixing_net.adj_action(action_probs,obs)
            return action_probs_fin
        else:
            return action_probs


    def update(self, state, obs, joint_action, action_prob,rewards, next_state, next_obs, dones):
        super(DQNLearner, self).update(state, obs, joint_action, action_prob,rewards, next_state, next_obs, dones)
        if self.warmup_phase <= 0:
#            print("========training DQN network=========")
            minibatch = self.memory.sample_batch(self.batch_size)
            minibatch_data = self.collect_minibatch_data(minibatch)
            histories = minibatch_data["pro_histories"]
            next_histories = minibatch_data["next_pro_histories"]
            actions = minibatch_data["pro_actions"]
            rewards = minibatch_data["pro_rewards"]
            self.update_step(histories, next_histories, actions, rewards, self.protagonist_optimizer)
            self.update_target_network()
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            self.training_count += 1
            return True
        return False

    def update_step(self, histories, next_histories, actions, rewards, optimizer):
        # print(histories.size())
        Q_values = self.policy_net(histories)

        Q_values = Q_values.gather(1, torch.tensor(actions, device=self.device, dtype=torch.long))
        # print(actions)
        # print(actions.size())

        # print(Q_values.size())
        # Q_values = self.policy_net(histories).gather(1, actions.unsqueeze(1)).squeeze()
        next_Q_values = self.target_net(next_histories)
        # next_Q_values = next_Q_values.view(next_histories.size(0), 3)
        next_Q_values = next_Q_values.max(1)[0].view(-1, 1)
        # next_Q_values = self.target_net(next_histories).max(1)[0].detach()
        # print("############",rewards.size())
        # print(rewards)

        target_Q_values = torch.tensor(rewards, device=self.device, dtype=torch.float32) + self.gamma * next_Q_values
        optimizer.zero_grad()
        # print(Q_values.size(),target_Q_values.size())
        loss = F.mse_loss(Q_values, target_Q_values)
        self.params["summary_write"].add_scalar("loss", loss, self.params["episode_num"])
        loss.backward()
        optimizer.step()
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return loss