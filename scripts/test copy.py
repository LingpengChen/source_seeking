import time
import sys


def print_progress_bar(index, iteration):
    position = index-1
    sys.stdout.write(f'\033[{position}B\r')  # Move to correct position
    sys.stdout.write(f'Case {index}: {iteration}\033[{position}A\r')  # Print progress bar and move back up
    sys.stdout.flush()

def main():
    total = 100
    num_bars = 3

    # 初始化，确保光标在最低点开始

    for i in range(total):
        time.sleep(1)  # 模拟工作
        # for j in range(1, num_bars + 1):
            # 更新每个进度条
        print_progress_bar(0, i)
        print_progress_bar(1, i)
        # move_cursor_up(num_bars)  # 将光标移回到第一个进度条的位置

    move_cursor_down(num_bars)  # 任务完成后，将光标移动到所有进度条下方

if __name__ == "__main__":
    main()
