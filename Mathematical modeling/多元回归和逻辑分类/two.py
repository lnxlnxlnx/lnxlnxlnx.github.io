import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# 数据准备
data = {
    '收入': [5, 10, 3, 8, 12],
    '债务': [2, 3, 4, 5, 2],
    '是否违约': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# 自变量和因变量
X = df[['收入', '债务']]
y = df['是否违约']

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