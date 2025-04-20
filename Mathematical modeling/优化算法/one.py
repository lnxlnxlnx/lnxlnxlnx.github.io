import numpy as np
from pulp import *

# 数据准备
weights = [2, 3, 5, 7]
values = [10, 15, 25, 35]
capacity = 15

# 模型构建
prob = LpProblem("Knapsack Problem", LpMaximize)
x = LpVariable.dicts("item", range(len(weights)), cat='Binary')

# 目标函数
prob += lpSum(values[i] * x[i] for i in range(len(weights)))

# 约束条件
prob += lpSum(weights[i] * x[i] for i in range(len(weights))) <= capacity

# 求解
prob.solve()

# 结果分析
print("状态:", LpStatus[prob.status])
print("总价值:", value(prob.objective))
for i in range(len(weights)):
    print(f"物品 {i+1} 是否选择: {x[i].varValue}")