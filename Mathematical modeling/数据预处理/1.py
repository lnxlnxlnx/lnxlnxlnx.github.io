import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 数据准备
time = np.array([0, 1, 2, 3, 4, 5, 6])
temperature = np.array([20, 22, np.nan, 25, 23, np.nan, 21])

# 去除缺失值
valid_indices = ~np.isnan(temperature)
time_valid = time[valid_indices]
temp_valid = temperature[valid_indices]

# 插值拟合
f = interp1d(time_valid, temp_valid, kind='quadratic', fill_value='extrapolate')
time_full = np.linspace(0, 6, 100)
temp_full = f(time_full)

# 结果分析
plt.plot(time, temperature, 'ro', label='原始数据')
plt.plot(time_full, temp_full, 'b-', label='插值拟合')
plt.xlabel('时间（小时）')
plt.ylabel('温度（°C）')
plt.legend()
plt.show()