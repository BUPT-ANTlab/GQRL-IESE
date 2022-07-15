from agent.controller import DeepLearningController, ReplayMemory
import random
import os
import numpy as np
from copy import deepcopy as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv



class mixing_net(torch.nn.Module):
    def __init__(self):
        super(mixing_net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_pur_net = nn.Sequential(
            nn.Conv1d(8, 2, kernel_size=4),
            # nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Flatten()
            # nn.Linear(36, 12),
            # nn.BatchNorm1d(12),
            # nn.ELU()
        )

        self.conv_eva_net = nn.Sequential(
            nn.Conv1d(8, 2, kernel_size=4),
            # nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Flatten()
            # nn.Linear(36, 12),
            # nn.BatchNorm1d(12),
            # nn.ELU()
        )

        self.traffic_flow_net = nn.Sequential(
            nn.Linear(48, 14),
            # nn.BatchNorm1d(12),
            nn.ELU()
        )
        #1*4*3
        self.Q_net = nn.Sequential(
            nn.Flatten()
            # nn.Linear(12, 12),
            # nn.BatchNorm1d(12),
            # nn.ELU()
        )
        #96
        self.out_net=nn.Sequential(
            nn.Linear(14*3+12, 1),
            # nn.BatchNorm1d(16),
            # nn.ELU(),
            # # nn.Linear(48, 16),
            # # nn.ELU(),
            # nn.Linear(16, 1)
        )

    def forward(self, total_state):
        pur_state_list=[]
        eva_state_list=[]
        traffic_flow_list=[]
        Q_list=[]
        for ele in (total_state):
            pur_state_list.append(ele["pur_state"])
            eva_state_list.append(ele["eva_state"])
            traffic_flow_list.append(ele["traffic_flow"])
            Q_list.append(ele["q"])
        pur_state=torch.tensor(pur_state_list, device=self.device, dtype=torch.float32)
        eva_state = torch.tensor(eva_state_list, device=self.device, dtype=torch.float32)
        traffic_flow = torch.tensor(traffic_flow_list, device=self.device, dtype=torch.float32)
        Q = torch.tensor(Q_list, device=self.device, dtype=torch.float32)
        pur_state=self.conv_pur_net(pur_state)
        eva_state=self.conv_eva_net(eva_state)
        traffic_flow=self.traffic_flow_net(traffic_flow)
        Q=self.Q_net(Q)
        state=torch.cat((pur_state, eva_state), -1)
        state=torch.cat((state, Q), -1)
        state = torch.cat((state, traffic_flow), -1)
        output = self.out_net(state)
        return output

class MemoryPool:
    def __init__(self, capacity=80000):
        self.memory = []
        self.capacity = capacity
        self.num_memory = 0

    def save(self, data):
        self.memory.append(data)
        self.num_memory = self.num_memory +1
        if self.num_memory > self.capacity:
            removed_transition = self.memory.pop(0)
            self.num_memory -= 1

    def sample_batch(self, minibatch_size):
        nr_episodes = self.size()
        if nr_episodes > minibatch_size:
            return random.sample(self.memory, minibatch_size)
        return self.memory

    def clear(self):
        self.memory.clear()
        self.num_memory = 0

    def size(self):
        return len(self.memory)

class MixingQ:
    def __init__(self,params):
        self.m_net=mixing_net().cuda(0)
        # self.m_net = mixing_net()
        self.load()
        self.best_train_loss = None
        self.memory = MemoryPool()
        self.train_batch_size=2048
        self.test_batch_size=256
        self.optimizer = torch.optim.Adam(self.m_net.parameters(), lr=1e-3)#
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss=None
        self.one_epoch_train_times=10
        self.test_times=10
        self.adj_loss=0.2
        self.train_times=0
        self.tor=0
        self.loss_path=os.path.join('mixing_net_model',"loss_log.csv")
        self.params=params

        # with open(self.loss_path, "w",newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["train_times", "train_loss","test_loss"])
        #     csvfile.close()






    # def predict(self,total_state):
    #     self.m_net.eval()
    #     return self.m_net(total_state)

    def train(self):
        if self.memory.num_memory > self.train_batch_size/2:
#            print("========training grad Q network=========")
            # self.m_net.train()
            all_loss=0
            #如果20步模型不变好，从保存的最好的模型开始训练
            if self.tor > 20:
                self.load()
            self.tor=self.tor+1
            self.train_times = self.train_times+1
            for i in range (self.one_epoch_train_times):
                train_data = self.memory.sample_batch(self.train_batch_size)
                self.optimizer.zero_grad()
                output=self.m_net(train_data)
                target_list=[]
                for ele in (train_data):
                    target_list.append(ele["total_reward"])
                target= torch.tensor(target_list, device=self.device, dtype=torch.float32).view(-1,1)
                loss_fn = torch.nn.MSELoss(reduction="mean")
                # print("target_size#####", target.size())
                # print("output_size#####", output.size())
                loss = loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                all_loss += loss.item()
            ave_train_loss=all_loss/self.train_times
            ave_test_loss=self.test()
            if self.best_loss is None:
                self.tor=0
                self.best_loss=ave_test_loss
                self.save_model()
            elif ave_test_loss<self.best_loss:
                self.tor = 0
                self.best_loss = ave_test_loss
                self.save_model()

            if self.best_train_loss is None:
                self.best_train_loss=ave_train_loss
            elif ave_train_loss<self.best_train_loss:
                self.best_train_loss=ave_train_loss

            with open(self.loss_path, "a+",newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.train_times, ave_train_loss,ave_test_loss])
                csvfile.close()
            if self.train_times%20==0:
                print("mixing network training loss is",ave_train_loss,"mixing network testing loss is",ave_test_loss, "mixing network best loss is",self.best_loss)
            return all_loss/self.train_times
        else:
            return None

    def test(self):
        all_loss=0
        # self.m_net.eval()
        for i in range (self.test_times):
            test_data = self.memory.sample_batch(self.test_batch_size)
            output = self.m_net(test_data)
            target_list = []
            for ele in (test_data):
                target_list.append(ele["total_reward"])
            target = torch.tensor(target_list, device=self.device, dtype=torch.float32).view(-1,1)
            # print("target_size%%%%%%%%%%%",target.size())
            # print("taget######",target[10])
            loss_fn = torch.nn.MSELoss(reduction="mean")
            loss = loss_fn(output, target)
            # print("output_size%%%%%%%%%",output.size())
            # print("output#####",output[10])
            all_loss += loss.item()
        ave_test_loss = all_loss /self.test_times
        return ave_test_loss

    def save_model(self):
        if not os.path.exists('./mixing_net_model'):
            os.mkdir('./mixing_net_model')
        torch.save(self.m_net, os.path.join('mixing_net_model', "mixing_model_all.pth"))
        torch.save(self.m_net.state_dict(), os.path.join('mixing_net_model',"mixing_model_stat_dic.pth"))


    def adj_action(self,action_probs,total_obs):

        if self.best_train_loss is not None:
            if (self.best_train_loss<10 and (total_obs[3] is not None)) or self.params.test:

                num_pur = len(action_probs)
                num_act = action_probs[0].size
                Q = np.zeros((num_pur, num_act))
                for i in range(num_pur):
                    index = np.argmax(action_probs[i])
                    Q[i][index] = 1

                obs = {
                    "pur_state": total_obs[3],
                    "eva_state": total_obs[1],
                    "q": Q,
                    "traffic_flow": total_obs[2]
                }
                score = self.m_net([obs]).detach().cpu().numpy()[0][0]

                for i in range(num_pur):
                    temp_Q=dc(Q)
                    temp_obs=dc(obs)
                    for j in range(num_act):
                        index = np.argmax(temp_Q[i])
                        temp_Q[i][index] = 0
                        temp_Q[i][j]=1
                        temp_obs['q']=temp_Q
                        temp_score=self.m_net([temp_obs]).detach().cpu().numpy()[0][0]
                        if temp_score>score:
                            score=temp_score
                            Q=dc(temp_Q)
                            obs=dc(temp_obs)
                final_Q=[]
                for i in range(num_pur):
                    final_Q.append(Q[i].reshape(num_act))
                return final_Q
            else:
                return action_probs
        else:
            return action_probs

    def save_exp(self,action_probs,total_obs,total_reward):
        if total_obs[3] is not None:
            obs={
                "pur_state": total_obs[3],
                "eva_state": total_obs[1],
                "q": action_probs,
                "traffic_flow": total_obs[2],
                "total_reward":total_reward
            }
            self.memory.save(obs)

    def load(self):
        model_patch=os.path.join('mixing_net_model',"mixing_model_stat_dic.pth")
        if os.path.exists(model_patch):
            # self.m_net.load_state_dict(torch.load(model_patch, map_location='cpu'))
            self.m_net.load_state_dict(torch.load(model_patch,  map_location='cuda'))


















