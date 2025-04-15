import numpy as np
from sklearn.linear_model import LinearRegression

# 数据
X = np.array([[100, 3, 5, 2], [120, 4, 8, 1], [80, 2, 3, 5], [150, 5, 10, 0.5], [90, 3, 4, 3]])
y = np.array([300, 350, 250, 400, 280])

# 模型训练
model = LinearRegression()
model.fit(X, y)

# 参数
beta_0 = model.intercept_
beta_1, beta_2, beta_3, beta_4 = model.coef_

print(f"截距 β0: {beta_0}")
print(f"系数 β1: {beta_1}, β2: {beta_2}, β3: {beta_3}, β4: {beta_4}")

new_house = np.array([[110, 3, 6, 1.5]])
predicted_price = model.predict(new_house)
print(f"预测房价: {predicted_price[0]} 万元")