"""
Testing out the n-armed bandit test bed. Final approach inspired by solutions produced in
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Bandit:
    def __init__(self, n=10, eps=.1, isTemp = False):
        self.n = n
        self.time = 0
        self.eps = eps
        self.isTemp = isTemp
        self.qTrue = []
        self.qEst = np.zeros(self.n)
        
        self.moveCount = []
        
        for i in range(0, n):
            self.qTrue.append(np.random.randn())
            self.qEst[i] = 0.
            self.moveCount.append(0)


    def getAction(self):
        prob = np.random.uniform(high=1, low=0, size=1)
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        reward = indices[0]
        if not self.isTemp:
            if prob > self.eps:
                reward = np.argmax(self.qEst)
        else:
            probEst = []
            for i in range(0, self.n):
                probEst.append(np.exp(self.qEst[i]/self.eps))
            probEst /= sum(probEst)
            eps = 0
            for i in range(0, self.n):
                eps += probEst[i]
                if prob < eps:
                    reward = i
        return reward


    def takeAction(self, At):
        Rt = self.qTrue[At] + np.random.randn()
        self.time += 1
        self.moveCount[At] += 1
        self.qEst[At] = self.qEst[At]*(self.moveCount[At]-1)/(self.moveCount[At]) + Rt/self.moveCount[At]
        return Rt


def banditSim(nBandits, bandits, nsteps):
    #Gets a list of bandit objects to simulate them for given number of steps
    avgRewards = np.zeros(nsteps, dtype=float)
    for i in range(0, nBandits):
        for t in range(0, nsteps):
            At = bandits[i].getAction()
            Rt = bandits[i].takeAction(At)
            avgRewards[t] += Rt
    avgRewards /= nBandits
    return avgRewards