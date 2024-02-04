import argparse
import time
import sys

def print_progress_bar(index, iteration):
    position = index+1
    sys.stdout.write(f'\033[{position}B\r')  # Move to correct position
    sys.stdout.write(f'Case {index}: {iteration}\033[{position}A\r')  # Print progress bar and move back up
    sys.stdout.flush()

def main(number):
    # 你的脚本逻辑
    for i in range(10):
        print_progress_bar(number, i)
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理一个数字参数')
    parser.add_argument('source_index', type=int, help='choose the sources topology you want')

    args = parser.parse_args()

    main(args.source_index)


