import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 数据准备
data = {
    '年龄': [30, 45, 50, 25, 60],
    '血压': [120, 130, 140, 110, 150],
    '是否患病': [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# 自变量和因变量
X = df[['年龄', '血压']]
y = df['是否患病']

# 模型拟合
model = LogisticRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 结果分析
print("准确率:", accuracy_score(y, y_pred))
print("混淆矩阵:\n", confusion_matrix(y, y_pred))
print("回归系数:", model.coef_)
print("截距:", model.intercept_)