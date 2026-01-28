
import json
import random
import os
from tqdm import tqdm

def load_id2meta(dataset):
    """Load ID to Metadata mapping"""
    path = f'{dataset}/sum_data/{dataset}_id2meta.json'
    print(f"Loading metadata from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_sasrec_sims(dataset):
    """Load SASRec similar items (numeric IDs)"""
    path = f'{dataset}/sum_data/similar_item_sasrec_num.txt'
    print(f"Loading SASRec similarities from {path}...")
    
    sims = {}
    with open(path, 'r', encoding='utf-8') as f:
        # Skip header
        f.readline()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            anchor = parts[0]
            # Store top 5 similar items for SFT to keep length reasonable
            # The file has 20, but we can't output all of them with meta info
            similar_items = parts[1:]
            sims[anchor] = similar_items
    return sims

def format_item_input(item_meta):
    """Format item information for input"""
    title = item_meta.get('title', '')
    description = item_meta.get('description', '')
    summary_words = item_meta.get('summary_words', [])
    
    # Text ID string: "[w1, w2, w3, w4, w5]"
    text_id = "[" + ", ".join(summary_words) + "]"
    
    return f"""Item text ID: {text_id} Title: {title}. Description: {description}. 
"""

def format_sim_item_output(item_meta):
    """Format similar item information for output"""
    title = item_meta.get('title', '')
    summary_words = item_meta.get('summary_words', [])
    text_id = "[" + ", ".join(summary_words) + "]"
    
    return f"Item text ID: {text_id} Title: {title}.\n"

def create_sft_sample(anchor_id, sim_ids, id2meta):
    """Create a single SFT training sample"""
    if anchor_id not in id2meta:
        return None
        
    anchor_meta = id2meta[anchor_id]
    
    # Filter valid similar items
    valid_sims = []
    for sim_id in sim_ids:
        if sim_id in id2meta:
            valid_sims.append(id2meta[sim_id])
            
    if not valid_sims:
        return None

    # Construct Prompt
    instruction = """Analyze the input product's information and identifiers. Based on collaborative filtering patterns (co-purchase or co-view signals), recommend similar products.
For each recommendation, provide its Title and Identifiers (5-word summary)."""

    input_text = f"""
Target Product:
{format_item_input(anchor_meta)}

Please recommend {len(valid_sims)} similar products:"""

    # Construct Output
    output_parts = []
    for i, sim_meta in enumerate(valid_sims, 1):
        output_parts.append(f"{i}. {format_sim_item_output(sim_meta)}")
    
    output_text = "\n\n".join(output_parts)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def main(dataset):
    id2meta = load_id2meta(dataset)
    sims = load_sasrec_sims(dataset)
    
    sft_data = []
    print(f"Generating SFT data for {dataset}...")
    
    for anchor, similar_list in tqdm(sims.items()):
        sample = create_sft_sample(anchor, similar_list, id2meta)
        if sample:
            sft_data.append(sample)
            
    # Save Data
    output_dir = "../LlamaFactory/data/grlm_in_domain"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/amazon_{dataset}_sasrec_collaborative20_sft.json"
    
    print(f"Saving {len(sft_data)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    datasets = ["beauty", "sports", "toys"]
    for ds in datasets:
        main(ds)
