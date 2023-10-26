# 假设 sample 是一个二维数组或一个包含列表的列表
sample = [[1, 2], [3, 4]]

# 假设 self.sent_samples 是一个包含列表的列表
sent_samples = []

# 对于 sample 中的每个元素（子列表），检查它是否在 self.sent_samples 中
for s in sample:
    if s in sent_samples:
        print(f"{s} is in self.sent_samples.")
    else:
        print(f"{s} is not in self.sent_samples.")
