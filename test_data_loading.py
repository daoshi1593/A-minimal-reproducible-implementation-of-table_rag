from utils.load_data import load_dataset

def main():
    dataset_path = "WikiTableQuestions/processed/processed_dataset.jsonl"
    
    # 测试加载前5个条目
    data = load_dataset("table_qa", dataset_path, stop_at=5)
    
    # 打印每个条目的基本信息
    for entry in data:
        print("\n=== Table Info ===")
        print(f"ID: {entry['id']}")
        print(f"问题: {entry.get('question', 'N/A')}")
        print(f"标签: {entry.get('label', 'N/A')}")
        print(f"Headers: {entry['headers']}")
        print("\nFirst few rows:")
        print(entry['table'].head(3))
        print("\n")

if __name__ == "__main__":
    main() 