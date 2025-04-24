import datasets

# File paths
L1_FILE = "L1_ChildrenStories.txt"
L2_FILE = "L2_BookCorpus.txt"
L3_FILE = "L3_CNN_DailyMail.txt"
L4_FILE = "L4_S2ORC.txt"

def process_dataset(dataset_name, output_file, text_keys, dataset_config=None):
    """Loads a dataset, extracts specified text fields, and writes up to 500MB of data to a file."""
    
    # Load dataset
    if dataset_config:
        dataset = datasets.load_dataset(dataset_name, dataset_config, split='train')
    else:
        dataset = datasets.load_dataset(dataset_name, split='train')
    
    # Print dataset details
    print(f"✅ Loaded {dataset_name} ({len(dataset)} records)")
    print(f"🔍 Sample keys: {dataset[0].keys()}")  # Check available keys

    # Check for missing fields
    missing_keys = [key for key in text_keys if key not in dataset[0]]
    if missing_keys:
        print(f"❌ Missing keys in dataset: {missing_keys}")
        return

    # Process and write data
    with open(output_file, "w", encoding="utf-8") as f:
        size = 0
        for item in dataset:
            text = " ".join([item[key] for key in text_keys if key in item])
            f.write(text + "\n")
            size += len(text.encode("utf-8"))
            if size >= 100 * 1024 * 1024:  # 500MB limit
                print(f"✅ Reached 500MB limit for {dataset_name}. Stopping...")
                break
    
    print(f"✅ Successfully processed {dataset_name}, saved to {output_file}")

# Process L1: Children Stories Collection
print("\n📖 Processing L1_ChildrenStories...")
process_dataset("ajibawa-2023/Children-Stories-Collection", L1_FILE, ["text"])

# Process L2: BookCorpus
print("\n📚 Processing L2_BookCorpus...")
process_dataset("bookcorpus", L2_FILE, ["text"])

# Process L3: CNN/DailyMail
print("\n📰 Processing L3_CNN_DailyMail...")
process_dataset("abisee/cnn_dailymail", L3_FILE, ["article", "highlights"], dataset_config="3.0.0")

# Process L4: S2ORC (Research Papers)
S2ORC_CONFIG = "ComputerScience,2019-2019"
print("\n📄 Processing L4_S2ORC...")
process_dataset("claran/modular-s2orc", L4_FILE, ["text"], dataset_config=S2ORC_CONFIG)  # Using 'text' instead of 'paper_text'

print("\n✅ All datasets processed successfully!")
