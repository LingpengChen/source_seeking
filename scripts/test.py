import argparse

def main(number):
    # 你的脚本逻辑
    print(f"处理数字: {number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理一个数字参数')
    parser.add_argument('source_index', type=int, help='choose the sources topology you want')

    args = parser.parse_args()

    main(args.source_index)
