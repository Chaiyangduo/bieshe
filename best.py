import numpy as np

# 读取 .npy 文件
data = np.load(r"C:\Users\22587\Desktop\SpeedWorld\controllers\supervisorGA_starter\Best.npy")

# 查看基本信息
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Array contents:")
print(data)

# # 如果数组太大，只查看部分内容
# print("\nFirst 5 elements:")
# print(data.flatten()[:5])  # 展平后查看前5个元素
#
# # 或者查看统计信息
# print("\nStatistics:")
# print("Min:", np.min(data))
# print("Max:", np.max(data))
# print("Mean:", np.mean(data))
# print("Std:", np.std(data))