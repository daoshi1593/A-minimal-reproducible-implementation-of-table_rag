import os
from utils.data_converter import convert_dataset

def main():
    # 设置数据目录
    data_dir = 'WikiTableQuestions'
    output_dir = 'WikiTableQuestions/processed'
    
    # 执行数据转换
    convert_dataset(data_dir, output_dir)
    
    print("数据集转换完成！")
    print(f"输出目录：{output_dir}")

if __name__ == '__main__':
    main() 