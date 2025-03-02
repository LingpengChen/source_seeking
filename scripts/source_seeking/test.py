import sys
import rospy

# 假设这是您的循环打印代码
for i in range(100):
    rospy.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 50)  # 清空当前行
    sys.stdout.write(f"\r当前值dfd df : {i}")  # 写入新内容
    sys.stdout.flush()