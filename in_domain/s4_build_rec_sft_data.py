import os
import json
import random
from tqdm import tqdm

def load_mapping_data(ckpt, dataset):
    """Load previously created mapping data"""
    # Load parent_asin to metadata mapping
    with open(f"./{dataset}/sum_data/{dataset}_id2meta.json", 'r', encoding='utf-8') as f:
        parent_asin2meta = json.load(f)
    
    # Load user interaction data
    sequential_file = f'./{dataset}/raw_data/sequential_data.txt'
    with open(sequential_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Sequential data lines: {len(lines)}")
    
    return parent_asin2meta, lines

def create_sft_data(parent_asin2meta, user_interactions, ckpt, dataset, min_items=6):
    """
    Create SFT training data - using all user item sequences
    
    Args:
        parent_asin2meta: ASIN to metadata mapping
        user_interactions: user interaction data
        min_items: minimum item count requirement (at least 5 input words + some output words)
    """
    
    sft_data = []
    skipped_users = 0
    total_sequences = 0
    item_id2tid = {}
    
    print("Starting to create SFT data...")
    
    for i, line in enumerate(user_interactions):
        line = line.strip()
        if not line:
            continue
        
        elements = line.split()
        if len(elements) <= 1:
            continue
        
        user_id = elements[0]
        item_ids = elements[1:]
        item_ids = item_ids[-20:] # Last 20 items to prevent too long sequences
        
        # Use all user items
        all_summary_words = []
        item_id_list = []
        meta_msg_list = []
        valid_sequence = True
        
        for item_id in item_ids:
            if item_id in parent_asin2meta:
                meta = parent_asin2meta[item_id]
                summary_words = meta.get('summary_words', [])
                if "" in summary_words:
                    valid_sequence = False
                    break
                # Filter invalid summary words, keep only non-empty ones
                valid_words = [word.replace("[","").replace("]","") for word in summary_words if word and word.strip()]
                if len(valid_words) >= 5:  # Need at least 3 valid words
                    all_summary_words.extend(valid_words[:5])  # Take at most 5 words per item
                    item_id_list.append(item_id)
                    meta_msg_list.append(meta)
                    item_id2tid[item_id] = valid_words[:5]
                else:
                    valid_sequence = False
                    break
            else:
                valid_sequence = False
                break
        
        if valid_sequence and len(all_summary_words) >= 15:  # Need at least 5 input words + 1 output word
            # Split input and output: first 5 words in input, rest in output
            input_words = all_summary_words[:5]  # First 5 words in input
            output_words = all_summary_words[5:]  # All remaining words in output
            test_ground_truth = output_words[-5:]
            valid_ground_truth = output_words[-10:-5]
            output_words = output_words[:-10]
            
            
            # Create instruction data
            sft_sample = create_instruction_sample(input_words, output_words, user_id, len(item_ids), item_id_list, meta_msg_list)
            sft_sample["item_id_list"] = item_id_list
            sft_sample["item_id_len"] = len(item_id_list)
            assert sft_sample["metadata"]["total_items"] == len(item_id_list)
            sft_sample["all_summary_words"] = all_summary_words
            sft_sample["valid_ground_truth_id"] = item_id_list[-2]
            sft_sample["test_ground_truth_id"] = item_id_list[-1]
            sft_sample["valid_ground_truth_tid"] = valid_ground_truth
            sft_sample["test_ground_truth_tid"] = test_ground_truth
            sft_sample["valid_ground_truth_msg"] = meta_msg_list[-2]
            sft_sample["test_ground_truth_msg"] = meta_msg_list[-1]
            sft_data.append(sft_sample)
            total_sequences += 1
    
    print(f"\nData statistics:")
    print(f"Total users: {len(user_interactions)}")
    print(f"Skipped users (sequence too short): {skipped_users}")
    print(f"Generated training samples: {total_sequences}")

    pat = f"./{dataset}/sum_data/item_id2tid"
    os.makedirs(pat, exist_ok=True)
    with open(f"{pat}/{dataset}_item_id2tid.json", 'w', encoding='utf-8') as f:
        json.dump(item_id2tid, f, ensure_ascii=False, indent=2)
    
    value2keys = {}
    for key, value in item_id2tid.items():
        value = ",".join(value)
        if value not in value2keys:
            value2keys[value] = []
        value2keys[value].append(key)
    
    with open(f"{pat}/{dataset}_tid2item_id.json", 'w', encoding='utf-8') as f:
        json.dump(value2keys, f, ensure_ascii=False, indent=2)
    
    print(f"item_id2tid count: {len(item_id2tid.keys())}")
    print(f"tid2item_id count: {len(value2keys.keys())}")
        
    return sft_data

def create_instruction_sample(input_words, output_words, user_id, total_items, item_id_list, meta_msg_list):
    """
    Create single instruction sample
    """
    # Instruction text - only describes the task
    instruction = "Based on the user's historical product interaction sequence, predict the next product's characteristic words. \nEach product is represented by exactly 5 characteristic words enclosed in square brackets []. The historical sequence shows the user's interaction pattern.\n"
    
    # Input text - first 5 words
    input_text = "Item text ID: [" + ", ".join(input_words) + "]"
    if "title" in meta_msg_list[0]:
        temp = meta_msg_list[0]["title"]
        input_text += f" Title: {temp}.\n"
    else:
        input_text += f" Title: None.\n"
    
    # Output text - all remaining words
    output_text = ""
    assert len(output_words)%5 == 0
    for i in range(total_items-3):
        output_text += "Item text ID: [" + ", ".join(meta_msg_list[i+1]["summary_words"]) + "]"
        if "title" in meta_msg_list[i+1]:
            temp = meta_msg_list[i+1]["title"]
            output_text += f" Title: {temp}.\n"
        else:
            output_text += f" Title: None.\n"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "user_id": user_id,
            "total_items": total_items,
            "total_words": len(input_words) + len(output_words),
            "input_word_count": len(input_words),
            "output_word_count": len(output_words)
        }
    }

def save_sft_data(sft_data, output_file):
    """Save SFT data"""
    print(f"\nSaving SFT data to: {output_file}")
    
    # Save complete data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    # Also save a simplified version (only contains instruction, input, output)
    simplified_data = []
    for sample in sft_data:
        simplified_data.append({
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": sample["output"]
        })
    
    simplified_file = output_file.replace('.json', '_simplified.json')
    with open(simplified_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)
    
    print(f"Simplified version saved to: {simplified_file}")

    
def main(ckpt, dataset):
    # Load data
    print(f"Loading {ckpt}/\/\{dataset} mapping data...")
    parent_asin2meta, user_interactions = load_mapping_data(ckpt, dataset)
    
    print(f"Loaded {len(parent_asin2meta)} product mappings")
    print(f"Loaded {len(user_interactions)} user interaction sequences")
    
    # Create SFT data - using all items
    sft_data = create_sft_data(
        parent_asin2meta, 
        user_interactions,
        ckpt, 
        dataset,
        min_items=2
    )
    
    # Save data
    output_file = f"../LlamaFactory/data/grlm_in_domain/amazon_{dataset}_sft_data_rec.json"
    save_sft_data(sft_data, output_file)
    print(f"\nSFT data creation completed! Generated {len(sft_data)} training samples")

if __name__ == "__main__":
    datasets = ["beauty", "sports", "toys"]
    ckpt = ""
    for dataset in datasets:
        main(ckpt, dataset)