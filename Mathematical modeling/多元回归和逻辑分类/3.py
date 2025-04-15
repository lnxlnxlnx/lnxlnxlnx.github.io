import numpy as np
from sklearn.linear_model import LinearRegression

# 数据
X = np.array([[10, 5, 50], [15, 8, 45], [8, 3, 60], [20, 10, 40], [12, 6, 55]])
y = np.array([200, 250, 150, 300, 220])

# 模型训练
model = LinearRegression()
model.fit(X, y)

# 参数
beta_0 = model.intercept_
beta_1, beta_2, beta_3 = model.coef_

print(f"截距 β0: {beta_0}")
print(f"系数 β1: {beta_1}, β2: {beta_2}, β3: {beta_3}")

new_product = np.array([[18, 7, 48]])   
predicted_sales = model.predict(new_product)
print(f"预测销量: {predicted_sales[0]} 件")