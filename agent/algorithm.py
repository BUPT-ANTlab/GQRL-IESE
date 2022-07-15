import torch
import agent.dqn as dqn
import agent.ppo as ppo
import agent.maddpg as maddpg
import agent.qmix as qmix


def make(algorithm, params):
    if algorithm == "DQN":
        return dqn.DQNLearner(params)

    if algorithm == "PPO":
        return ppo.PPOLearner(params)

    if algorithm == "MADDPG":
        return maddpg.MADDPGLearner(params)

    if algorithm == "QMIX":
        return qmix.QMIXLearner(params)

