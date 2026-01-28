import os
import json
import random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
import re

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list

def load_similarities(similarity_file):
    with open(similarity_file, 'r') as f:
        similarities = json.load(f)
    return similarities

def prepare_prompt(item, top_similar_items, all_items_dict):
    """Prepare prompt including information from the 5 most similar items"""
    
    # Current item information
    title = item.get('title', '')
    if title != "":
        title_text = f"Title: {title}"
    else:
        title_text = ""
    
    description = item.get('description', '')
    if description != "":
        if len(description) > 150:
            description_text = f"Description: {description[:150]}..."
        else:
            description_text = f"Description: {description}"
    else:
        description_text = ""
    
    # Prepare similar items information
    similar_items_info = []
    for similar_item in top_similar_items:
        similar_item_id = similar_item['item_id']
        similarity_score = similar_item['similarity']
        
        if similar_item_id in all_items_dict:
            similar_item_data = all_items_dict[similar_item_id]
            similar_title = similar_item_data.get('title', '')
            similar_desc = similar_item_data.get('description', '')
            
            similar_info = f"Similar Item {similar_item_id} (similarity: {similarity_score:.3f}):"
            if similar_title:
                similar_info += f" Title: {similar_title}"
            if similar_desc: 
                if len(similar_desc) > 150:
                    similar_info += f" Description: {similar_desc[:150]}..."
                else:
                    description_text = f"Description: {similar_desc}"
            similar_items_info.append(similar_info)
    
    similar_items_text = "\n".join(similar_items_info)
    
    # Prompt
    prompt = f"""You are an expert product summarizer. Your task is to generate exactly FIVE words to summarize this product. Please follow ALL guidelines carefully:

GUIDELINES:
1. WORD FORM: All words must be in their base form (nouns or adjectives, no -ed, -ing, -s endings)
2. WORD ORDER: Order words by importance (most important aspect first)
3. CONTENT FOCUS: Focus on these aspects in order:
   a) Main product category/type (e.g., "doll", "puzzle", "car")
   b) Key function or purpose (e.g., "educational", "remote-control")
   c) Distinctive features (e.g., "wooden", "electronic", "collectible")
   d) Target audience (e.g., "toddler", "boys", "family")
   e) Unique selling point (e.g., "glow-in-dark", "interactive")
4. CONSISTENCY WITH SIMILAR ITEMS: Consider the similar items provided. If they share common characteristics, use consistent terminology for those aspects.
5. UNIQUENESS: Include at least 1-2 words that distinguish this product from the similar items. Each product should have some unique aspects.
6. OUTPUT FORMAT: Provide ONLY the five words in this exact format: [word1, word2, word3, word4, word5]
7. NO ADDITIONAL TEXT: Do not include any explanations, thoughts, or other content.

PRODUCT INFORMATION:
{title_text}
{description_text}

TOP 5 SIMILAR PRODUCTS (for reference):
{similar_items_text}

ANALYSIS GUIDANCE:
1. First, identify what this product has in common with similar products (shared category, features, audience)
2. Then, identify what makes this product unique or different
3. Use consistent vocabulary for shared characteristics
4. Include distinctive vocabulary for unique aspects
5. Ensure words cover the five required aspects in order

Please provide exactly five words in this exact format: [word1, word2, word3, word4, word5]:"""

    return prompt

def process_batch_on_gpu(rank, data_slice, output_queue, model_name, 
                         similarities_dict, all_items_dict, batch_size=1):
    print(f"Rank {rank}: Initializing model...")
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Rank {rank}: Starting to process {len(data_slice)} items")
    
    results = []
    
    for i in tqdm(range(0, len(data_slice), batch_size), desc=f"Rank {rank}"):
        batch_items = data_slice[i:i + batch_size]
        batch_results = process_single_batch(batch_items, model, tokenizer, device, 
                                            similarities_dict, all_items_dict)
        results.extend(batch_results)
    
    output_queue.put((rank, results))
    print(f"Rank {rank}: Processing completed with {len(results)} items")

def process_single_batch(items, model, tokenizer, device, similarities_dict, all_items_dict):
    prompts = []
    for item in items:
        item_id = item['id']
        
        top_similar_items = []
        if item_id in similarities_dict:
            similar_items = similarities_dict[item_id]
            top_similar_items = similar_items[:5]
        
        prompt = prepare_prompt(item, top_similar_items, all_items_dict)
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(text)
    
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_attention_mask=True,
        max_length=32768
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    
    results = []
    for i, (item, input_ids, output_ids) in enumerate(zip(items, model_inputs.input_ids, generated_ids)):
        generated_output_ids = output_ids[len(input_ids):].tolist()
        
        content = tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip("\n")
        
        item_id = item['id']
        similar_item_ids = []
        if item_id in similarities_dict:
            similar_items = similarities_dict[item_id][:5]
            similar_item_ids = [sim_item['item_id'] for sim_item in similar_items]

        words = []
        if content:
            # Extract [word1, word2, word3, word4, word5] format
            pattern = r'\[([^\]]+)\]'
            match = re.search(pattern, content)
            
            if match:
                inner_content = match.group(1)
                if ',' in inner_content:
                    words = [word.strip().lower().strip('"\'\[\]') for word in inner_content.split(',')]
                else:
                    words = [word.strip().lower().strip('"\'\[\]') for word in inner_content.split()]
            else:
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        inner = line[1:-1]
                        if ',' in inner:
                            words = [word.strip().lower().strip('"\'') for word in inner.split(',')]
                        else:
                            words = [word.strip().lower().strip('"\'') for word in inner.split()]
                        break
                if not words:
                    if ',' in content:
                        words = [word.strip().lower().strip('"\'\[\]') for word in content.split(',')]
                    else:
                        words = [word.strip().lower().strip('"\'\[\]') for word in content.split()]
            
            words = words[:5]
            while len(words) < 5:
                words.append("")
        
        item_copy = item.copy()
        item_copy['llm_output'] = content
        item_copy['summary_words'] = words
        item_copy['similar_item_ids'] = similar_item_ids
        
        results.append(item_copy)
    
    return results

def analyze_statistics_with_similarity(all_items, similarities_dict):
    print("\n" + "="*50)
    print("Statistical Analysis Results")
    print("="*50)
    
    all_words = []
    word_freq = Counter()
    word_by_position = [Counter() for _ in range(5)]
    
    for item in all_items:
        words = item.get('summary_words', [])
        all_words.extend([word for word in words if word])
        
        for i, word in enumerate(words):
            if i < 5 and word:
                word_by_position[i][word] += 1
    
    word_freq.update(all_words)
    
    print(f"\n1. Overall Vocabulary Statistics:")
    print(f"   Total words: {len(all_words)}")
    print(f"   Unique words: {len(word_freq)}")
    print(f"   Top 20 most frequent words:")
    for word, count in word_freq.most_common(20):
        print(f"     {word}: {count}")
    
    print(f"\n2. Vocabulary Statistics by Position:")
    positions = ['Product Category', 'Function/Purpose', 'Features', 'Audience', 'Unique Point']
    for i, (pos, counter) in enumerate(zip(positions, word_by_position)):
        print(f"   Top 10 words for {pos} position:")
        for word, count in counter.most_common(10):
            print(f"     {word}: {count}")
    
    print(f"\n3. Conflict Analysis:")
    summary_tuples = [tuple(item.get('summary_words', [])) for item in all_items]
    tuple_counter = Counter(summary_tuples)
    
    duplicate_tuples = [(tup, count) for tup, count in tuple_counter.items() if count > 1]
    total_conflicts = sum(count - 1 for tup, count in duplicate_tuples)
    conflict_rate = total_conflicts / len(all_items) if all_items else 0
    
    print(f"   Identical summaries count: {len(duplicate_tuples)}")
    print(f"   Conflicting items count: {total_conflicts}")
    print(f"   Conflict rate: {conflict_rate:.4f}")
    
    if duplicate_tuples:
        print(f"   Top 5 most frequent conflicts:")
        for tup, count in sorted(duplicate_tuples, key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {tup}: appears {count} times")
    
    print(f"\n4. Validity Check:")
    valid_items = 0
    partial_items = 0
    
    for item in all_items:
        words = item.get('summary_words', [])
        valid_words = [word for word in words if word]
        if len(valid_words) == 5:
            valid_items += 1
        elif len(valid_words) > 0:
            partial_items += 1
            print(item)
    
    invalid_items = len(all_items) - valid_items - partial_items
    
    print(f"   Complete 5-word items: {valid_items}/{len(all_items)} ({valid_items/len(all_items)*100:.2f}%)")
    print(f"   Partially valid items: {partial_items}/{len(all_items)} ({partial_items/len(all_items)*100:.2f}%)")
    print(f"   Invalid items: {invalid_items}/{len(all_items)} ({invalid_items/len(all_items)*100:.2f}%)")
    
    print(f"\n5. Similar Item Vocabulary Consistency Analysis:")
    
    item_dict = {item['id']: item for item in all_items}
    
    similarity_scores = []
    shared_word_counts = []
    
    for item in all_items:
        item_id = item['id']
        item_words = set([w for w in item.get('summary_words', []) if w])
        
        if item_id in similarities_dict:
            similar_items = similarities_dict[item_id][:5]
            
            for sim_item in similar_items:
                sim_id = sim_item['item_id']
                similarity = sim_item['similarity']
                
                if sim_id in item_dict:
                    sim_item_data = item_dict[sim_id]
                    sim_words = set([w for w in sim_item_data.get('summary_words', []) if w])
                    
                    shared_words = item_words.intersection(sim_words)
                    shared_word_counts.append(len(shared_words))
                    similarity_scores.append(similarity)
    
    if similarity_scores:
        avg_similarity = np.mean(similarity_scores)
        avg_shared_words = np.mean(shared_word_counts)
        
        print(f"   Average similarity: {avg_similarity:.4f}")
        print(f"   Average shared words: {avg_shared_words:.2f}")
        
        if len(similarity_scores) > 1:
            correlation = np.corrcoef(similarity_scores, shared_word_counts)[0, 1]
            print(f"   Correlation between similarity and shared words: {correlation:.4f}")
    
    return {
        'word_frequency': dict(word_freq.most_common()),
        'position_frequency': [dict(counter.most_common()) for counter in word_by_position],
        'conflict_analysis': {
            'total_conflicts': total_conflicts,
            'conflict_rate': conflict_rate,
            'duplicate_tuples': [(list(tup), count) for tup, count in duplicate_tuples]
        },
        'validity_analysis': {
            'valid_items': valid_items,
            'partial_items': partial_items,
            'invalid_items': invalid_items
        },
        'similarity_analysis': {
            'avg_similarity': avg_similarity if 'avg_similarity' in locals() else 0,
            'avg_shared_words': avg_shared_words if 'avg_shared_words' in locals() else 0,
            'correlation': correlation if 'correlation' in locals() else 0
        }
    }

def main(dataset):
    model_name = "../hf_qwen3_2507_4b"
    num_gpus = 8
    
    data_file = f"./{dataset}/raw_data/{dataset}.item.json"
    dataset_similarities_file = f"./{dataset}/sum_data/{dataset}_similarities.json"
    
    print(f"Loading data: {data_file}")
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")
    
    print(f"Loading similarity data: {dataset}_similarities_file")
    similarities_dict = load_similarities(dataset_similarities_file)
    print(f"Loaded similarity information for {len(similarities_dict)} items")
    
    all_items_dict = {item['id']: item for item in data}
    
    chunk_size = len(data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + chunk_size
        data_chunks.append(data[start_idx:end_idx])
    
    print(f"Dataset split into {num_gpus} chunks, chunk sizes: {[len(chunk) for chunk in data_chunks]}")
    
    processes = []
    output_queue = mp.Queue()
    
    print("Starting multi-process processing...")
    start_time = time.time()
    
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(rank, data_chunks[rank], output_queue, model_name, 
                  similarities_dict, all_items_dict, 1)
        )
        processes.append(p)
        p.start()
    
    all_results = []
    for _ in range(num_gpus):
        rank, results = output_queue.get()
        print(f"Received {len(results)} results from Rank {rank}")
        all_results.extend(results)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f"Multi-process processing completed, total time: {end_time - start_time:.2f} seconds")
    
    # Save results
    output_file = f"./{dataset}/sum_data/{dataset}_summaries_with_similarity.jsonl"
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistical analysis
    stats = analyze_statistics_with_similarity(all_results, similarities_dict)
    
    # Save statistics
    stats_file = f"./{dataset}/sum_data/{dataset}_statistics_with_similarity.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")
    
    # Output examples
    print(f"\nExample outputs (first 3 items):")
    for i, item in enumerate(all_results[:3]):
        print(f"\nItem ID: {item['id']}")
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Summary words: {item.get('summary_words', [])}")
        if 'similar_item_ids' in item:
            print(f"Referenced similar items: {item['similar_item_ids']}")
    
    print(f"\nProcessing completed! Total items processed: {len(all_results)}")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    datasets = ["beauty", "sports", "toys"]
    for dataset in datasets:
        main(dataset)