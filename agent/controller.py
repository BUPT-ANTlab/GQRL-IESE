import random
import numpy
import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity):
        self.transitions = []
        self.capacity = capacity
        self.nr_transitions = 0

    def save(self, transition):
        self.transitions.append(transition)
        self.nr_transitions += len(transition[0])
        if self.nr_transitions > self.capacity:
            removed_transition = self.transitions.pop(0)
            self.nr_transitions -= len(removed_transition[0])

    def sample_batch(self, minibatch_size):
        nr_episodes = self.size()
        # print(self.size())
        if nr_episodes > minibatch_size:
            # print("################")
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()
        self.nr_transitions = 0

    def size(self):
        return len(self.transitions)


class Controller:

    def __init__(self, params):
        self.params = params
        self.num_agents = params["num_pursuit"]
        self.num_actions = params["num_action"]
        self.actions = list(range(self.num_actions))
        self.agent_ids = list(range(self.num_agents))
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.env = params["env"]

    def policy(self, observations, mixing_net,training_mode=True):
        random_joint_action = [random.choice(self.actions) for _ in self.num_agents]
        return random_joint_action

    def update(self, state, observations, joint_action, action_prob,rewards, next_state, next_observations, dones):
        return True


class DeepLearningController(Controller):

    def __init__(self, params):
        super(DeepLearningController, self).__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.use_global_reward = params["use_global_reward"]
        self.input_ego_state_shape = [8, 10]
        self.input_oth_state_shape = [8, 10]
        self.input_eva_state_shape = [8, 10]
        self.global_input_shape = params["global_observation_shape"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.warmup_phase = params["warmup_phase"]
        self.episode_transitions = []
        self.target_update_period = params["target_update_period"]
        self.epsilon = 1
        self.training_count = 0
        self.current_histories = []
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.policy_net = None
        self.target_net = None
        self.default_observations = [numpy.zeros((8,10))]
        self.max_history_length = 1

    def save_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.save_weights(path)

    def load_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.load_weights(path)

    def save_weights_to_path(self, path):
        if self.policy_net is not None:
            self.policy_net.save_weights_to_path(path)

    def load_weights_from_history(self, path):
        if self.policy_net is not None:
            self.policy_net.load_weights_from_history(path)

    def policy(self, observations, mixing_net, training_mode=True):
        new_observations = self.change_observation(observations)
        action_probs = self.joint_action_probs(new_observations, observations, mixing_net, training_mode)
        return action_probs, [numpy.random.choice(self.actions, p=probs) for probs in action_probs]

    def joint_action_probs(self, histories, observations, mixing_net, training_mode=True, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.agent_ids
        return [numpy.ones(self.num_actions) / self.num_actions for _ in agent_ids]

    def update(self, state, observations, joint_action, action_prob,rewards, next_state, next_observations, dones):
        observations = self.change_observation(observations)
        next_observations = self.change_observation(next_observations)
        return self.update_transition(state, observations, joint_action, action_prob,rewards, next_state, next_observations, dones)

    def update_transition(self, state, obs, joint_action, action_prob,rewards, next_state, next_obs, dones):
        self.warmup_phase = max(0, self.warmup_phase - 1)
        if self.use_global_reward:
            global_reward = sum(rewards)
            rewards = [global_reward for _ in range(self.num_agents)]
        pro_obs = []
        pro_actions = []
        next_pro_obs = []
        pro_rewards = []
        for i in range(self.num_agents):
            pro_obs.append(obs[i])
            pro_actions.append(joint_action[i])
            next_pro_obs.append(next_obs[i])
            pro_rewards.append(rewards[i])
        protagonist_ids = [i for i in self.agent_ids]
        # pro_probs = self.joint_action_probs(pro_obs, state,training_mode=True, agent_ids=protagonist_ids,)
        # state = [temp[1:] if type(temp[0]) == str else temp[:] for temp in state]
        # next_state = [temp[1:] if type(temp[0]) == str else temp[:] for temp in next_state]
        self.episode_transitions.append((obs, pro_obs, pro_actions, action_prob, pro_rewards, next_state, next_pro_obs))
        # global_terminal_reached = not [d for i, d in enumerate(dones) if (not d)]
        global_terminal_reached = dones
        if global_terminal_reached:
            s, pro_obs, a1, p1, pro_rewards, sn, next_pro_obs = tuple(zip(*self.episode_transitions))
            R1 = self.to_returns(pro_rewards, protagonist_ids)
            self.memory.save((s, pro_obs, a1, p1, pro_rewards, sn, next_pro_obs, R1))
            self.episode_transitions.clear()
            self.current_histories.clear()
        return True

    def collect_minibatch_data(self, minibatch, whole_batch=False):
        states=[]
        states_all = []


        actions=[]

        action_probs=[]

        rewards=[]

        next_state_all = []
        next_states=[]
        rewards_all=[]



        for episode in minibatch:
            states_, pro_obs, p_actions, pro_probs, p_rewards, next_states_, next_pro_obs, p_R = episode
            states.extend(list(np.array(pro_obs).reshape(-1,3,8,10)))
            states_all.extend(list((np.array(pro_obs)[:,0,:,:])))
            actions.extend(list(np.array(p_actions).reshape(-1,1)))
            action_probs.extend(list(np.array(pro_probs).reshape(-1,3)))
            rewards.extend(list(np.array(p_rewards).reshape(-1,1)))
            next_states.extend(list(np.array(next_pro_obs).reshape(-1,3,8,10)))
            next_state_all.extend(list((np.array(next_pro_obs)[:,0,:,:])))
            reward_sum=np.array(p_rewards)[:,0]+np.array(p_rewards)[:,1]+np.array(p_rewards)[:,2]+np.array(p_rewards)[:,3]
            rewards_all.extend(list(reward_sum.reshape(-1, 1)))

            # print("$$$$$$$$$$$$$$$$$$$$$")
            # print(np.array(p_rewards).shape)
            # print(np.array(np.array(p_rewards)[:,0]+np.array(p_rewards)[:,1]+np.array(p_rewards)[:,2]+np.array(p_rewards)[:,3]).shape)
            # print("$$$$$$$$$$$$$$$$$$$$$")

        return {
            "pro_histories": np.array(states),
            "pro_actions": np.array(actions),
            "pro_action_probs": np.array(action_probs),
            "pro_rewards": np.array(rewards),
            "next_pro_histories": np.array(next_states),
            "pro_rewards_all": np.array(rewards_all),
            "next_pro_histories_all":np.array(next_state_all),
            "pro_histories_all": np.array(states_all)
        }

        # pro_histories_numpy = np.array(pro_histories)
        # pro_actions_numpy = np.array(pro_actions)
        # pro_action_probs_numpy = np.array(pro_action_probs)
        # pro_rewards_numpy = np.array(pro_rewards)
        # next_pro_histories_numpy = np.array(next_pro_histories)
        # pro_returns_numpy = np.array(pro_returns)

        # return {
        #         "pro_histories": torch.tensor(pro_histories_numpy, device=self.device, dtype=torch.float32), \
        #         "pro_actions": torch.tensor(pro_actions_numpy, device=self.device, dtype=torch.long), \
        #         "pro_action_probs": torch.tensor(pro_action_probs_numpy, device=self.device, dtype=torch.float32), \
        #         "pro_rewards": torch.tensor(pro_rewards_numpy, device=self.device, dtype=torch.float32), \
        #         "next_pro_histories": torch.tensor(next_pro_histories_numpy, device=self.device, dtype=torch.float32), \
        #         "pro_returns": torch.tensor(pro_returns_numpy, device=self.device, dtype=torch.float32)
        #         }
                
#        return {
#                "pro_histories": torch.tensor(pro_histories, device=self.device, dtype=torch.float32), \
#                "pro_actions": torch.tensor(pro_actions, device=self.device, dtype=torch.long), \
#                "pro_action_probs": torch.tensor(pro_action_probs, device=self.device, dtype=torch.float32), \
#                "pro_rewards": torch.tensor(pro_rewards, device=self.device, dtype=torch.float32), \
#                "next_pro_histories": torch.tensor(next_pro_histories, device=self.device, dtype=torch.float32), \
#                }
#
    def reshape_histories(self, history_batch):
        histories = []
        for i in range(1):
            joint_observations = []
            for joint_history in history_batch:
                joint_observations += joint_history[i]
            histories.append(joint_observations)
        return histories

    def to_returns(self, individual_rewards, agent_ids):
        R = numpy.zeros(len(agent_ids))
        discounted_returns = []
        for rewards in reversed(individual_rewards):
            R = numpy.array(rewards) + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return numpy.array(discounted_returns)

    def normalized_returns(self, discounted_returns):
        R_mean = numpy.mean(discounted_returns)
        R_std = numpy.std(discounted_returns)
        return (discounted_returns - R_mean) / (R_std + self.eps)

    def update_target_network(self):
        target_net_available = self.target_net is not None
        if target_net_available and self.training_count % self.target_update_period is 0:
            self.target_net.protagonist_net.load_state_dict(self.policy_net.protagonist_net.state_dict())
            self.target_net.protagonist_net.eval()

    def change_observation(self, observations):
        total_obs=[]
        for pursuit_id in self.params["pursuit_ids"]:
            single_obs=[]
            single_obs.append(observations[0][pursuit_id]["ego_pos"])
            single_obs.append(observations[0][pursuit_id]["oth_pos"])
            single_obs.append(observations[1])
            total_obs.append(single_obs)
        return total_obs

        # if self.params["no_task"]:
        #     total_observation = []
        #     signal_observation = []
        #     for pursuit_id in self.params["pursuit_ids"]:
        #         for observation in observations:
        #             if observation[0] == pursuit_id:
        #                 signal_observation = [observation[1:]]
        #
        #         for last_pursuit_id in range(self.params["num_pursuit"] - len(signal_observation)):
        #             signal_observation.append([0 for _ in range(self.params["local_observation_shape"][1])])
        #
        #         for observation in observations:
        #             if type(observation[0]) == str and observation[0] not in self.params["pursuit_ids"]:
        #                 signal_observation.append(observation[1:])
        #
        #         for last_evader_id in range(self.params["num_evader"]-(len(signal_observation) - 4)):
        #             signal_observation.append([0 for _ in range(self.params["local_observation_shape"][1])])
        #
        #         total_observation.append(signal_observation)
        #     return total_observation
        # else:
        #     total_observation = []
        #     for pursuit_id in self.params["pursuit_ids"]:
        #         for observation in observations:
        #             if observation[0] == pursuit_id:
        #                 signal_observation = [observation[1:]]
        #
        #         target_eva = self.params["env"].pursuit_vehs[pursuit_id]["target_evader"]
        #         pursuit_eva_list = []
        #         for temp_pursuit_id in self.params["env"].pursuit_vehs.keys():
        #             if self.params["env"].pursuit_vehs[temp_pursuit_id]["target_evader"] == target_eva:
        #                 pursuit_eva_list.append(temp_pursuit_id)
        #         for observation in observations:
        #             if observation[0] in pursuit_eva_list and observation[0] != pursuit_id:
        #                 signal_observation.append(observation[1:])
        #         for last_pursuit_id in range(self.params["num_pursuit"]-len(signal_observation)):
        #             signal_observation.append([0 for _ in range(self.params["local_observation_shape"][1])])
        #         for observation in observations:
        #             if observation[0] == target_eva:
        #                 signal_observation.append(observation[1:])
        #         for last_evader_id in range(self.params["num_evader"]-(len(signal_observation) - 4)):
        #             signal_observation.append([0 for _ in range(self.params["local_observation_shape"][1])])
        #
        #         total_observation.append(signal_observation)
        #     return total_observation


