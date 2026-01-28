import json
import random
from tqdm import tqdm

def load_mapping_data(dataset):
    """Load previously created mapping data"""
    # Load parent_asin to metadata mapping
    with open(f'./{dataset}/sum_data/{dataset}_id2meta.json', 'r', encoding='utf-8') as f:
        parent_asin2meta = json.load(f)
    return parent_asin2meta

def prepare_data(item):
    data = {}
    title = item.get('title', '')
    if title != "":
        title = f"Title: {title}"
    description = item.get('description', '')
    if description != "":
        description = f"Description: {description}"
    
    # Prompt
    data["instruction"] = """Please generate exactly five words to summarize this product. Follow these guidelines carefully:

1. Words must be in their base form (noun or adjective, no -ed, -ing, -s endings)
2. Order words by importance (most important aspect first)
3. Focus on product category, function, key features, and target users
4. Each word should represent a distinct aspect
5. The word should be able to express the uniqueness of the product to ensure that it is distinguishable from other similar products
6. Provide ONLY the five words in the specified format, with no additional text, explanations, or content
7. Output format: [word1, word2, word3, word4, word5]"""
    data["input"] = f"""\n\nProduct Information:\n{title}\n{description}\n\nPlease provide exactly five words separated by commas:"""
    data["output"] = "[" + ", ".join(item.get('summary_words', '')) + "]"

    return data

def create_sft_data(parent_asin2meta):
    """
    Create SFT training data - using all user item sequences
    
    Args:
        parent_asin2meta: ASIN to metadata mapping
        user_interactions: user interaction data
        min_items: minimum item count requirement (at least 5 input words + some output words)
    """
    
    sft_data = []
    
    print("Starting to create SFT data...")
    
    for key, value in parent_asin2meta.items():
        sft_data.append(prepare_data(value))
        
    return sft_data

def save_sft_data(sft_data, output_file):
    """Save SFT data"""
    print(f"\nSaving SFT data to: {output_file}")
    
    # Save complete data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

def main(dataset):
    # Load data
    print("Loading mapping data...")
    parent_asin2meta = load_mapping_data(dataset)
    
    print(f"Loaded {len(parent_asin2meta)} product mappings")
    
    # Create SFT data - using all items
    sft_data = create_sft_data(
        parent_asin2meta, 
    )
    
    # Save data
    output_file = f"../LlamaFactory/data/grlm_in_domain/amazon_{dataset}_sft_data_meta2tid.json"
    save_sft_data(sft_data, output_file)

    print(f"\nSFT data creation completed! Generated {len(sft_data)} training samples")

if __name__ == "__main__":
    datasets = ["beauty", "sports", "toys"]
    for dataset in datasets:
        main(dataset)