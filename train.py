import os.path
from os.path import join
import csv
import numpy as np
import torch
import numpy
import random
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def run_episode(episode_id, controller, params, QO_net,training_mode=True):
    env = params["env"]
    path = params["directory"]
    num_pursuit = params["num_pursuit"]
    time_step = 0
    pursuit_discounted_return = 0
    pursuit_undiscounted_return = 0
    last_observation_state = None
    last_joint_action = None
    last_action_prob=None
    last_global_state = None
    finally_epoch = 0
    for epoch in range(params["Epoch"]):
        stop_, rewards = env.step()
        # if params["algorithm_name"] == "DQN" or params["algorithm_name"] == "PPO" or params["algorithm_name"] == "MADDPG":
        pursuit_undiscounted_return += sum(rewards)
        pursuit_discounted_return += (params["gamma"]**time_step)*sum(rewards)
        time_step += 1
        # if epoch > params["Epoch"] - 1 or stop_:
        if epoch > params["Epoch"] - 1:
            env.reset()
            break
        else:
            if len(env.pursuit_vehs.keys()) != params["num_pursuit"]:
                observation_state = env.observation_state
                # global_state = env.global_state
                env.pursuitVehControl(choice_random=True)
                env.evadeVehControl(choice_random=True)
            else:
                observation_state = env.observation_state
                # global_state = env.global_state
                policy_updated = False
                if training_mode and last_observation_state is not None:
                    # print("===========training===================")
                    policy_updated = controller.update(last_observation_state, last_observation_state, last_joint_action,last_action_prob,
                                                       rewards, observation_state, observation_state, stop_)
                    QO_net.save_exp(last_action_prob,last_observation_state,sum(rewards))
                    QO_net.train()

                if stop_:
                    env.reset()
                    finally_epoch = epoch
                    break
                action_prob,joint_action = controller.policy(observation_state, QO_net,training_mode)
                # joint_action = np.array([0, 0, 0, 0])
                joint_action = np.array(joint_action)
                env.pursuitVehControl(commands=joint_action)
                env.evadeVehControl(choice_random=True)
                last_observation_state = observation_state
                # last_global_state = global_state
                last_joint_action = joint_action
                last_action_prob=action_prob

    return pursuit_discounted_return/time_step, pursuit_undiscounted_return/time_step, policy_updated, finally_epoch


def run_test_record(controller, params,mixing_net):
    env = params["env"]
    path = params["directory"]
    num_pursuit = params["num_pursuit"]
    time_step = 0
    pursuit_discounted_return = 0
    pursuit_undiscounted_return = 0
    pursuit_discounted_records = []
    pursuit_undiscounted_records = []
    for epoch in range(params["Epoch"]):
        stop_, rewards = env.step()
        # if params["algorithm_name"] == "DQN" or params["algorithm_name"] == "PPO" or params["algorithm_name"] == "MADDPG":
        pursuit_undiscounted_return += sum(rewards)
        pursuit_undiscounted_records.append(sum(rewards))
        pursuit_discounted_return += (params["gamma"] ** time_step) * sum(rewards)
        pursuit_discounted_records.append((params["gamma"] ** time_step) * sum(rewards))
        time_step += 1
        if epoch > params["Epoch"] - 1:
            env.reset()
            break
        else:
            if len(env.pursuit_vehs.keys()) != params["num_pursuit"]:
                observation_state = env.observation_state
                # global_state = env.global_state
                env.pursuitVehControl(choice_random=True)
                env.evadeVehControl(choice_random=True)
            else:
                observation_state = env.observation_state
                # global_state = env.global_state
                if stop_:
                    env.reset()
                    break
                action_probs,joint_action = controller.policy(observation_state, mixing_net,training_mode=False)
                # joint_action = np.array([0, 0, 0, 0])
                joint_action = np.array(joint_action)
                env.pursuitVehControl(commands=joint_action)
                env.evadeVehControl(choice_random=True)

    return pursuit_discounted_return / time_step, pursuit_undiscounted_return / time_step,\
           pursuit_discounted_records, pursuit_undiscounted_records, time_step


def run_test(num_test_episodes, controller, params,QO_net):
    num_pursuit = params["num_pursuit"]
    test_discounted_returns = []
    test_undiscounted_returns = []
    finally_epochs = []
    for test_episode_id in range(num_test_episodes):
        print("============> test in episode %s  <============" % (test_episode_id+1))
        discounted_returns, undiscounted_returns, updated, finally_epoch = run_episode(episode_id="Test-{}".format(test_episode_id),
                                                                        controller=controller, params=params,
                                                                        training_mode=False,QO_net=QO_net)
        finally_epochs.append(finally_epoch)
        test_discounted_returns.append(discounted_returns)
        test_undiscounted_returns.append(undiscounted_returns)
    return np.mean(test_discounted_returns), np.mean(test_undiscounted_returns), np.mean(finally_epochs)


def run(controller, params,QO_net=None):
    path = params["directory"]
    env = params["env"]
    summary_write = SummaryWriter(path)
    params["summary_write"] = summary_write
    num_test_episodes = params["test_episodes"]
    training_discounted_returns = []
    training_undiscounted_returns = []
    test_discounted_returns = []
    test_undiscounted_returns = []
    test_discounted_return, test_undiscounted_return, finally_epochs = run_test(num_test_episodes, controller, params,QO_net)
    test_discounted_returns.append(test_discounted_return)
    test_undiscounted_returns.append(test_undiscounted_return)
    epoch_updates = 0
    best_finally_epoch = 7200
    best_discounted_return = -1000
    for episode in range(params["Episode"]):
        params["episode_num"] = episode
        print("============> run in episode %s <============" % episode)
        discounted_returns, undiscounted_returns, updated, finally_epoch = run_episode(episode, controller, params,QO_net=QO_net)
        training_discounted_returns.append(discounted_returns)
        training_undiscounted_returns.append(undiscounted_returns)
        if updated:
            print("============> updated in episode", epoch_updates, "<============")
            if epoch_updates % 1 == 0:
                test_discounted_return, test_undiscounted_return, finally_epochs = run_test(num_test_episodes, controller, params,QO_net)
                test_discounted_returns.append(test_discounted_return)
                test_undiscounted_returns.append(test_undiscounted_return)
                summary_write.add_scalar("test_discounted_return", test_discounted_return, int(epoch_updates/1))
                summary_write.add_scalar("test_undiscounted_return", test_undiscounted_return, int(epoch_updates/1))
                # if test_discounted_return > best_discounted_return:
                #     print("Save the best example, epoch is %s" % epoch_updates)
                #     # best_finally_epoch = finally_epoch
                #     best_discounted_return = test_discounted_return
                #     best_path = join(path, "best.pth")
                #     controller.save_weights_to_path(best_path)
            if epoch_updates % 20 == 0:
                # if best_finally_epoch > finally_epoch:
                if discounted_returns > best_discounted_return:
                    print("Save the best example, epoch is %s" % epoch_updates)
                    # best_finally_epoch = finally_epoch
                    best_discounted_return = discounted_returns
                    best_path = join(path, "best.pth")
                    controller.save_weights_to_path(best_path)
            if epoch_updates % 30 == 1:
                if os.path.exists(join(path, "best.pth")):
                    controller.load_weights_from_history(join(path, "best.pth"))
            epoch_updates += 1
        summary_write.add_scalar("discounted_return", discounted_returns, episode)
        summary_write.add_scalar("undiscounted_return", undiscounted_returns, episode)
    return True


def test(controller, params, testEpoch,mixing_net=None):
    temp = 0
    for test_epoch in range(testEpoch):
        discounted_return, undiscounted_return, discounted_records, undiscounted_records, step = run_test_record(controller, params,mixing_net)
        print("Undiscounted return", undiscounted_return, "Step", step)
        with open(params["test_output_csv"], "a+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, undiscounted_return])
            csvfile.close()

        # if undiscounted_return > temp:
        #     temp = undiscounted_return
        #     record = np.array(undiscounted_records)
        #     data1 = pd.DataFrame(record)
        #     data1.to_csv(params["test_output_csv"])


        # if step > 194 and undiscounted_return > 0:
        #     temp = undiscounted_return
        #     record = np.array(undiscounted_records)
        #     data1 = pd.DataFrame(record)
        #     data1.to_csv(params["test_output_csv"])

