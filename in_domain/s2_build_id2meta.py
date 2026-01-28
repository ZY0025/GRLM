import json
from tqdm import tqdm
from collections import Counter, defaultdict

def create_simple_mapping(dataset):
    """Create simple parent_asin mapping"""
    input_file = f"./{dataset}/sum_data/{dataset}_summaries_with_similarity.jsonl"
    output_file = f"./{dataset}/sum_data/{dataset}_id2meta.json"
    
    all_items = []
    parent_asin2meta = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing data"):
            if line.strip():
                item = json.loads(line)
                if len(item["summary_words"]) != 5:
                    print(f"line:{line} len(summary_words) ≠ 5")
                item["summary_words"] = ["-".join(word.split()) for word in item["summary_words"]]
                all_items.append(item)
                parent_asin = item.get('id')
                if parent_asin:
                    parent_asin2meta[parent_asin] = item
    
    # Save mapping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parent_asin2meta, f, ensure_ascii=False, indent=2)
    
    print(f"Completed! Processed {len(parent_asin2meta)} products")
    return parent_asin2meta

# Example usage function
def query_product_info(parent_asin2meta, asin):
    """Query product information for a specific ASIN"""
    if asin in parent_asin2meta:
        meta = parent_asin2meta[asin]
        return meta
    else:
        return None

if __name__ == "__main__":
    datasets = ["beauty", "sports", "toys"]
    for dataset in datasets:
        # Create mapping
        mapping = create_simple_mapping(dataset)
        
        # Example query
        sample_asin = list(mapping.keys())[0] if mapping else None
        if sample_asin:
            info = query_product_info(mapping, sample_asin)
            print(f"Example query - ASIN: {sample_asin}")
            print(f"Product information: {info}")