import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 数据准备
data = {
    '年龄': [25, 30, 35, 40, 45],
    '收入': [5, 8, 10, 12, 15],
    '工作满意度': [7, 6, 8, 9, 7]
}
df = pd.DataFrame(data)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df)

# 结果分析
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA 结果')
plt.show()
print("解释的方差比例:", pca.explained_variance_ratio_)