my_list = [[3, 3], [5, 6], [3, 3]]

# 将内部列表转换成元组
tuple_list = [tuple(item) for item in my_list]

# 使用集合去除重复元素，并转换回列表
unique_list = [list(item) for item in set(tuple_list)]

print(unique_list)  # 输出: [[3, 3], [5, 6]] 或者 [[5, 6], [3, 3]]，取决于集合操作的结果
