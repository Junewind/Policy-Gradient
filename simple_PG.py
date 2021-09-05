#!/usr/bin/env python
# coding: utf8
# @Time    : 2021/9/5 下午9:17
# @Author  : Yichuan
# @Email   : z_yichuan@163.com
# @Software: PyCharm


import numpy as np
import gym
import torch.nn as nn
import torch as t
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

criterion = nn.CrossEntropyLoss(reduction='none')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dicount_factor = 0.99
eplison = 0.1  # 增加动作选择的随机性
lr = 0.02
env = gym.make("CartPole-v0")
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped
batch_size = 1
epochs = 1000


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()

        # 下面定义两个全连接层就可以了
        self.linear1 = nn.Linear(4, 10)
        nn.init.normal_(self.linear1.weight, 0, 0.3)
        nn.init.constant_(self.linear1.bias, 0.1)
        self.linear2 = nn.Linear(10, 2)
        nn.init.normal_(self.linear2.weight, 0, 0.3)
        nn.init.constant_(self.linear2.bias, 0.1)

    def forward(self, x):
        out = t.from_numpy(x).float()

        out = self.linear1(out)
        out = F.tanh(out)

        out = self.linear2(out)
        prob = F.softmax(out, dim=1)  # 这个输出主要是用来使用概率来挑选动作

        return prob, out


def choose_action(prob):
    action = np.random.choice(a=2, p=prob[0].detach().numpy())

    return action


def get_one_batch(agent):
    reward_an_episode = []
    observation_an_episode = []
    action_an_episode = []
    observation = env.reset()
    done = False

    while not done:
        env.render()
        observation = np.expand_dims(observation, axis=0)
        prob, log_prob = agent(observation)
        observation_an_episode.append(observation)
        action = choose_action(prob)
        action_an_episode.append(action)
        observation, reward, done, info = env.step(action)

        reward_an_episode.append(reward)

    return action_an_episode, np.concatenate(observation_an_episode, axis=0), reward_an_episode


def learn():
    # 定义一个网络实例
    agent = Agent()

    train_loss = []
    train_reward = []
    for e in range(epochs):

        # 定义一个优化器
        optim = t.optim.Adam(agent.parameters(), lr=lr)
        batch_data = get_one_batch(agent)

        # 下面开始计算损失函数，要注意，这里的损失函数是有agent所获得奖励的来的
        # 先计算奖励的累计
        agent.train()

        actions = t.tensor(batch_data[0])
        observations = batch_data[1]
        rewards = batch_data[2]
        train_reward.append(sum(rewards))

        acc_reward = []
        for i in range(len(rewards)):
            acc_r = 0
            for j in range(i, len(rewards)):
                acc_r += dicount_factor ** (j - i) * rewards[j]
            acc_reward.append(acc_r)
        acc_reward = t.tensor(acc_reward)
        acc_reward -= acc_reward.mean()
        acc_reward /= acc_reward.std()

        prob, logits = agent(observations)
        log_prob = criterion(logits, actions)
        log_reward = log_prob * acc_reward

        loss = log_reward.mean()

        train_loss.append(loss)
        optim.zero_grad()

        loss.backward()
        optim.step()

    plt.plot(train_loss)
    plt.plot(train_reward)


if __name__ == '__main__':
    learn()
    env.close()


