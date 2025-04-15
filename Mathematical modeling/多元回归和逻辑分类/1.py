import numpy as np
from sklearn.linear_model import LinearRegression

# 数据
X = np.array([[20, 5, 5], [25, 3, 8], [15, 8, 3], [30, 2, 10], [18, 6, 4]])
y = np.array([85, 90, 75, 95, 80])

# 模型训练
model = LinearRegression()
model.fit(X, y)

# 参数
beta_0 = model.intercept_
beta_1, beta_2, beta_3 = model.coef_

print(f"截距 β0: {beta_0}")
print(f"系数 β1: {beta_1}, β2: {beta_2}, β3: {beta_3}")

new_student = np.array([[22, 4, 6]])
predicted_score = model.predict(new_student)
print(f"预测成绩: {predicted_score[0]} 分")