import numpy as np
import matplotlib.pyplot as plt
import pywt

# 数据准备
time = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * time) + 0.5 * np.random.randn(100)

# 小波去噪
wavelet = 'db4'
level = 1
coeffs = pywt.wavedec(signal, wavelet, level=level)
threshold = 0.5 * np.std(coeffs[-level])
coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if i >= -level else c for i, c in enumerate(coeffs)]
signal_denoised = pywt.waverec(coeffs_thresh, wavelet)

# 结果分析
plt.plot(time, signal, 'b-', label='原始信号')
plt.plot(time, signal_denoised, 'r-', label='去噪信号')
plt.xlabel('时间（秒）')
plt.ylabel('信号值')
plt.legend()
plt.show()