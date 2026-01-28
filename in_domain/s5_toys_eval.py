from collections import defaultdict
import re
import os
import json
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_reverse_mapping(original_dict):
    """Create reverse mapping, split key into word list"""
    reverse_mapping = {}
    word_to_keys = defaultdict(list)
    
    for key_str, ids in original_dict.items():
        # Clean and split keywords
        words = [word.strip().lower() for word in key_str.split(',')]
        reverse_mapping[key_str] = {
            'words': words,
            'ids': ids
        }
        
        # Build index for each word
        for word in words:
            word_to_keys[word].append(key_str)
    
    return reverse_mapping, word_to_keys


def get_iid_by_tid(content):
    threshold = 0
    iids = []
    tids = content.replace("[","").replace("]","").split(", ")
    tid_key = ",".join(tids)
    if tid_key in tid2item_id:
        iids.extend(tid2item_id[tid_key])
    else:
        # return []
        # Need fuzzy matching
        candidate_scores = defaultdict(float)
        query_words = tids
        for i, query_word in enumerate(query_words):
            # Position weight: words at front are more important
            position_weight = 1.0 / (i + 1)  # First word weight 1.0, second 0.5, third 0.33...
            
            # Find candidates containing current query word
            for candidate_word, candidate_keys in word_to_keys.items():
                # Calculate similarity (simple containment relationship)
                similarity = 0.0
                if query_word == candidate_word:
                    similarity = 1.0  # Exact match
                elif query_word in candidate_word or candidate_word in query_word:
                    similarity = 0.8  # Partial match
                # If similarity exceeds threshold, add score to all related candidates
                if similarity > 0:
                    for candidate_key in candidate_keys:
                        candidate_scores[candidate_key] += similarity * position_weight
        
        # Sort by score and filter
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply threshold
        matched_results = []
        for candidate_key, score in sorted_candidates:
            if score >= threshold:
                matched_results.append({
                    'key': candidate_key,
                    'score': score,
                    'ids': reverse_mapping[candidate_key]['ids']
                })
            iids.extend(reverse_mapping[candidate_key]['ids'])
        
        iids = iids[:1]
    return iids

def extend_iid_by_tid(content, t):
    threshold = 0
    iids = []
    tids = content.replace("[","").replace("]","").split(", ")

    # Need fuzzy matching
    candidate_scores = defaultdict(float)
    query_words = tids
    for i, query_word in enumerate(query_words):
        # Position weight: words at front are more important
        position_weight = 1.0 / (i + 1)  # First word weight 1.0, second 0.5, third 0.33...
        
        # Find candidates containing current query word
        for candidate_word, candidate_keys in word_to_keys.items():
            # Calculate similarity (simple containment relationship)
            similarity = 0.0
            if query_word == candidate_word:
                similarity = 1.0  # Exact match
            elif query_word in candidate_word or candidate_word in query_word:
                similarity = 0.8  # Partial match
            # If similarity exceeds threshold, add score to all related candidates
            if similarity > 0:
                for candidate_key in candidate_keys:
                    candidate_scores[candidate_key] += similarity * position_weight
    
    # Sort by score and filter
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Apply threshold
    matched_results = []
    for candidate_key, score in sorted_candidates:
        if score >= threshold:
            matched_results.append({
                'key': candidate_key,
                'score': score,
                'ids': reverse_mapping[candidate_key]['ids']
            })
        iids.extend(reverse_mapping[candidate_key]['ids'])
        
    iids = iids[:1]
    return iids

def load_test_data(file_path):
    """Load test data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    return sft_data

def prepare_batch_prompts(batch_data):
    """Prepare prompts for batch data"""
    batch_prompts = []
    batch_metadata = []  # Store metadata for each sample
    
    for d in batch_data:
        # l = d["input"] + ", " + d["output"] + ", [" + ", ".join(d["valid_ground_truth_tid"]) + "]"
        l = d["input"] + d["output"] + "Item text ID: [" + ", ".join(d["valid_ground_truth_tid"]) + "]"
        if "title" in d["valid_ground_truth_msg"]:
            temp = d["valid_ground_truth_msg"]["title"]
            l += f" Title: {temp}.\n"
        else:
            l += f" Title: None.\n"
        prompt = "Based on the user's historical product interaction sequence, predict the next product's characteristic words. \n" + \
            "Each product is represented by exactly 5 characteristic words enclosed in square brackets []. The historical sequence shows the user's interaction pattern.\n"
        prompt += l
        prompt += "Item text ID: "


        messages = [{"role": "user", "content": prompt}]
        batch_prompts.append(messages)
        batch_metadata.append({
            'original_data': d,
            'iid_gt': d["test_ground_truth_id"],
            'tid_gt': d["test_ground_truth_tid"],
            'prompt': prompt
        })
    
    return batch_prompts, batch_metadata

def process_single_gpu(rank, data_slice, output_queue, model_name, total_items, batch_size=8):
    """Process data slice on each GPU with batch testing"""
    print(f"Rank {rank}: Initializing model and tokenizer...")
    
    # Set GPU for current process
    torch.cuda.set_device(rank)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set left padding (important!)
    tokenizer.padding_tide = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to specific GPU
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype=torch.float16,
        device_map=f"cuda:{rank}"
    )
    model.eval()
    
    print(f"Rank {rank}: Model loaded, starting to process {len(data_slice)} test samples, batch_size={batch_size}")
    
    local_score = [0] * 20
    local_results = []
    
    # Process data in batches
    for batch_start in tqdm(range(0, len(data_slice), batch_size), desc=f"GPU {rank}"):
        batch_end = min(batch_start + batch_size, len(data_slice))
        batch_data = data_slice[batch_start:batch_end]
        
        # Prepare batch prompts
        batch_prompts, batch_metadata = prepare_batch_prompts(batch_data)
        
        # Apply chat template in batch
        batch_texts = []
        for messages in batch_prompts:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_texts.append(text)
        
        # Batch encoding with left padding
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768,  # Adjust according to your model
            return_attention_mask=True
        ).to(model.device)

        # Batch text generation
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=30,
                do_sample=False,
                num_beams=20,
                num_return_sequences=20,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False,
            )
        
        # Process batch generation results
        batch_results = process_batch_results(
            generated_ids, model_inputs, batch_metadata, tokenizer
        )
        
        local_results.extend(batch_results)
        
        # Update scores
        for result in batch_results:
            for i, iid in enumerate(result["iids"]):
                if i < len(local_score) and result["iid_gt"] == iid:
                    local_score[i] += 1
                    break
    
    # Put results in queue
    output_queue.put((rank, local_score, local_results))
    print(f"Rank {rank}: Processing completed, processed {len(local_results)} samples in total")

def process_batch_results(generated_ids, model_inputs, batch_metadata, tokenizer):
    """Process batch generation results"""
    batch_results = []
    
    # Calculate number of generated sequences per sample
    num_sequences_per_sample = generated_ids.shape[0] // len(batch_metadata)
    
    for batch_idx, metadata in enumerate(batch_metadata):
        dic = metadata['original_data'].copy()
        iid_gt = metadata['iid_gt']
        
        all_results = []
        contents = []
        raw_contents = []
        
        # Extract all generated sequences for current sample
        start_idx = batch_idx * num_sequences_per_sample
        end_idx = (batch_idx + 1) * num_sequences_per_sample
        
        for seq_idx in range(start_idx, end_idx):
            input_len = model_inputs.input_ids[batch_idx].shape[0]
            output_ids = generated_ids[seq_idx][input_len:].tolist()
            
            # Parse thinking content
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            raw_content = content
            raw_contents.append(raw_content)

            pattern = r'\[(.*?)\]'
            cons = re.findall(pattern, content)
            for c in cons:
                content_str = "[" + c + "]"
                if content_str not in contents:
                    contents.append(content_str)
        
        dic["contents_len"] = len(contents)

        iids = []
        for i, content in enumerate(contents):
            iid = get_iid_by_tid(content)
            all_results.append({
                'sequence_id': i,
                'content': content,
                'iid': iid
            })
            iids.extend(iid)
        
        # Remove duplicates
        ids = []
        for i in iids:
            if i not in ids:
                ids.append(i)
        
        iids = ids[:20]
        
        t = 30
        while len(iids) < 20:
            for i, content in enumerate(contents):
                iid = extend_iid_by_tid(content, t)
                all_results.append({
                    'sequence_id': i,
                    'content': content,
                    'iid': iid
                })
                for single_iid in iid:
                    if single_iid not in iids:
                        iids.append(single_iid)
                    if len(iids) >= 20:
                        break
                if len(iids) >= 20:
                    break
            
            if t >= 100:
                break
            
            t += 20
        
        iids = iids[:20]

        dic['prompt'] = metadata['prompt']
        dic['raw_contents'] = raw_contents
        dic["all_results"] = all_results
        dic["iids"] = iids
        dic["iids_len"] = len(iids)
        dic["iid_gt"] = iid_gt
        
        batch_results.append(dic)
    
    return batch_results

def calculate_recall(scores, total_samples):
    """Calculate recall metrics"""
    recall_metrics = {}
    recall_metrics["recall@1"] = sum(scores[:1]) / total_samples
    recall_metrics["recall@5"] = sum(scores[:5]) / total_samples
    recall_metrics["recall@10"] = sum(scores[:10]) / total_samples
    recall_metrics["recall@20"] = sum(scores[:20]) / total_samples
    return recall_metrics

tid2item_id_path = './toys/sum_data/item_id2tid/toys_tid2item_id.json'

with open(tid2item_id_path, 'r', encoding='utf-8') as f:
    tid2item_id = json.load(f)

reverse_mapping, word_to_keys = create_reverse_mapping(tid2item_id)

def main():
    # Model paths
    ckpts = [
        "../sft_ckpts/grlm_indomain_toys/",
        ]

    for ckpt in ckpts:
        model_name = f"{ckpt}"
        
        # Test data path
        test_file = "../LlamaFactory/data/grlm_in_domain/amazon_toys_sft_data_rec.json"

        # Load test data
        print(f"Loading test data: {test_file}")
        sft_data = load_test_data(test_file)
        print(f"Loaded {len(sft_data)} test samples")
        
        # Distribute data based on GPU count
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPUs")
        
        # Split data into chunks
        chunk_size = len(sft_data) // num_gpus
        data_chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            if i == num_gpus - 1:  # Last chunk contains all remaining data
                end_idx = len(sft_data)
            else:
                end_idx = start_idx + chunk_size
            data_chunks.append(sft_data[start_idx:end_idx])
        
        print(f"Data split into {num_gpus} chunks, chunk sizes: {[len(chunk) for chunk in data_chunks]}")
        
        # Create multi-processes
        processes = []
        output_queue = mp.Queue()
        
        print("Starting multi-process testing...")
        start_time = time.time()
        
        # Set batch size, can adjust based on your GPU memory
        batch_size = 1  # Decrease this value if memory is insufficient
        
        for rank in range(num_gpus):
            p = mp.Process(
                target=process_single_gpu,
                args=(rank, data_chunks[rank], output_queue, model_name, len(sft_data), batch_size)
            )
            processes.append(p)
            p.start()
        
        # Collect results
        all_results = []
        all_scores = [0] * 20
        
        for _ in range(num_gpus):
            rank, local_score, local_results = output_queue.get()
            print(f"Received {len(local_results)} results from GPU {rank}")
            
            # Merge scores
            for i in range(min(len(local_score), len(all_scores))):
                all_scores[i] += local_score[i]
            
            all_results.extend(local_results)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        end_time = time.time()
        print(f"Multi-GPU testing completed, total time: {end_time - start_time:.2f} seconds")
        
        # Calculate recall metrics
        recall_metrics = calculate_recall(all_scores, len(sft_data))
        
        # Output recall results
        print("\n" + "="*50)
        print("Test Results")
        print("="*50)
        for metric, value in recall_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save detailed results
        bac = ckpt.replace("/","_")
        output_file = f"./toys/rec_res/toys_eval_info.json"
        print(f"\nSaving detailed results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Save recall results
        recall_file = f"./toys/rec_res/toys_recall_results.json"
        with open(recall_file, 'w', encoding='utf-8') as f:
            json.dump(recall_metrics, f, ensure_ascii=False, indent=2)
        print(f"Recall results saved to: {recall_file}")
        
        print(f"\nTesting completed! Total samples processed: {len(all_results)}")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()