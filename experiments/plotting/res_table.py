import pickle
import numpy as np
from pkg_resources import resource_filename

BASE_PATH = resource_filename("abcrl", "/..")


for task, lengths in zip(["IMDb", "tldr", "openllama"], [101, 201, 201]):
    with open(f"{BASE_PATH}/results/numerics/{task}.pkl", "rb") as f:
        if task == "IMDb":
            res = pickle.load(f)
        else:
            res, _ = pickle.load(f)

    for method in ["abc", "rlhf", "uniform"]:
        rewards = []
        for key, value in res.items():
            if method.lower() == key.split("_")[0]:
                rewards.append(value[:lengths])
        rewards = np.array(rewards)
        # print(f"{task} - {method}: {rewards.shape}")
        average_rewards = np.mean(rewards, axis=1)
        # print(f"{task} - {method}: {average_rewards.shape}")
        n = len(average_rewards)

        mean = np.mean(average_rewards)
        std = np.std(average_rewards)
        print(f"{task} - {method}: {mean} +- {2* std / np.sqrt(n)}")
        # print(task, method, mean.mean(), mean.std())
