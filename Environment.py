import numpy as np
import pandas as pd
from Actor_Critic import ActorCritic

if __name__ == '__main__':
    data1 = pd.read_csv(r"...\daily_adjusted_AAPL.csv")
    #print(data)
    df1 = data1["adjusted_close"][::-1]
    df1.index = df1.index[::-1]
    df1 = df1.to_numpy()
    data2 = pd.read_csv(r"...\daily_IBM.csv")
    df2 = data2["adjusted_close"][::-1]
    df2.index = df2.index[::-1]
    df2 = df2.to_numpy()

    def rewardfunction(self, action, resultbefore, resultnow):
        diff = resultnow - resultbefore
        if diff >0:
            if action == 0:
                reward = 0
            else:
                reward = -diff
        else:
            if action == 1:
                reward = 0
            else:
                reward = diff
        return reward

    model = ActorCritic(inputdims=60, actiondims=2, layers=4, neurons=120)

    x = df2
    episodes = 20
    zeroc=0
    days=60
    for e in range(0,episodes+1):
        totalreward = 0
        for i in range(days, len(x)):
            state = x[i-days:i]
            eps = 1
            state = (state - np.mean(state)) / (np.std(state) + eps)
            state_ = x[i-days+1:i+1]
            eps = 1
            state_ = (state_ - np.mean(state_)) / (np.std(state_) + eps)
            action, action_prob= model.choose_action(state)
            resultbefore = x[i-1]
            resultnow = x[i]
            reward = rewardfunction(action, resultbefore, resultnow)
            model.store_transition(state, action, reward, state_)
            totalreward += reward
        model.learn()
        
        if totalreward==0:
            if zeroc == 10:
                break
            zeroc+=1
        if totalreward != 0:
            zeroc = 0
        print("Totalreward :", totalreward, "episode :", e)