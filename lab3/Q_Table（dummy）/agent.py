import random

import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma # 衰减系数
        self.epsilon = 0
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        self.sample_count += 1
        if self.sample_count < 10:
            self.epsilon = 1
        else: self.epsilon = 1 - np.exp(self.sample_count * 0.018)
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.action_dim)  #随机探索选取一个动作
        else: action = self.predict(state)

        return action

    def update(self, state, action, reward, next_state, done):
        ############################ Q表格的更新方法（需要完成）##################################
        if done:
            self.Q_table[state] = reward
        else:
            self.Q_table[state,action] += self.lr * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

    def predict(self, state):

        m = np.max(self.Q_table[state])
        actions = np.where(self.Q_table[state] >= m)[0]
        return np.random.choice(actions)

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
