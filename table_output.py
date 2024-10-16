import torch
import os 
import numpy as np 
import pandas as pd

path_u = "Result/Multiplicative_incre_2step_CR_unit.pkl"
path_c = "Result/Multiplicative_incre_2step_CR_cluster.pkl"


result_u = torch.load(path_u)
result_c = torch.load(path_c)

print(len(result_u[result_u[:,0]>0]))
print(len(result_c[result_c[:,0]>0]))

# true_gate = 2.489
true_gate = 2

# unit
bias = result_u.mean(axis=0) - true_gate
std = result_u.std(axis=0) 
mse = bias **2 + std ** 2

print("unit")
df = pd.DataFrame({"bias":bias, "std":std, "mse":mse})
for row_ in df.iterrows():
    row = row_[1]
    print("{:.3f} & {:.3f} & {:.3f}".format(row["bias"], row["std"], row["mse"]))

print("\n")


print("cluster")
# cluster
bias = result_c.mean(axis=0) - true_gate
std = result_c.std(axis=0) 
mse = bias **2 + std ** 2

df = pd.DataFrame({"bias":bias, "std":std, "mse":mse})
for row_ in df.iterrows():
    row = row_[1]
    print("{:.3f} & {:.3f} & {:.3f}".format(row["bias"], row["std"], row["mse"]))


