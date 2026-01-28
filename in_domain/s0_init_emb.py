import os
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data(file_path: str) -> List[Dict]:
    """Load JSON file and convert to list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list

def prepare_text_for_embedding(item: Dict) -> str:
    """Prepare text for embedding generation"""
    title = item.get('title', '')
    description = item.get('description', '')
    categories = item.get('categories', '')
    
    text_parts = []
    if title:
        text_parts.append(f"Title: {title}")
    if description:
        text_parts.append(f"Description: {description}")
    if categories:
        text_parts.append(f"Categories: {categories}")
    
    return " | ".join(text_parts)

def process_batch_on_gpu(rank: int, data_slice: List[Dict], output_queue: mp.Queue, 
                         model_name: str, batch_size: int = 16):
    """Process data slice on specific GPU"""
    print(f"Rank {rank}: Initializing model...")
    
    # Set GPU for current process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device: {device}")
    
    # Load model and tokenizer
    print(f"Rank {rank}: Loading model and tokenizer...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Prepare texts
    texts = [prepare_text_for_embedding(item) for item in data_slice]
    
    all_embeddings = []
    all_results = []
    
    print(f"Rank {rank}: Generating embeddings for {len(texts)} items...")
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Rank {rank}"):
        batch_texts = texts[i:i + batch_size]
        batch_items = data_slice[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs[0]
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 normalization
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings_np)
            
            # Save results
            for j, (item, embedding) in enumerate(zip(batch_items, embeddings_np)):
                item_copy = item.copy()
                item_copy['embedding'] = embedding.tolist()
                all_results.append(item_copy)
    
    # Combine all embeddings
    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
    else:
        embeddings_array = np.array([])
    
    end_time = time.time()
    print(f"Rank {rank}: Embeddings finished in {end_time - start_time:.2f}s")
    
    # Put results in queue
    output_queue.put((rank, all_results, embeddings_array))

def generate_embeddings_multi_gpu(data: List[Dict], model_name: str, 
                                  num_gpus: int = None, batch_size: int = 16) -> Tuple[List[Dict], np.ndarray]:
    """Generate embeddings using multiple GPUs"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"Using {num_gpus} GPUs for parallel processing")
    print(f"Total items: {len(data)}")
    
    # Split data into chunks
    chunk_size = len(data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + chunk_size
        data_chunks.append(data[start_idx:end_idx])
    
    print(f"Data split into {num_gpus} chunks, sizes: {[len(chunk) for chunk in data_chunks]}")
    
    # Create processes
    processes = []
    output_queue = mp.Queue()
    
    print("Starting multi-process processing...")
    start_time = time.time()
    
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(rank, data_chunks[rank], output_queue, model_name, batch_size)
        )
        processes.append(p)
        p.start()
    
    # Collect results
    all_results = []
    all_embeddings = []
    
    for _ in range(num_gpus):
        rank, results, embeddings = output_queue.get()
        print(f"Received {len(results)} results from Rank {rank}")
        all_results.extend(results)
        if embeddings.size > 0:
            all_embeddings.append(embeddings)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Combine all embeddings
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
    else:
        combined_embeddings = np.array([])
    
    end_time = time.time()
    print(f"\nMulti-GPU processing completed in {end_time - start_time:.2f}s")
    
    return all_results, combined_embeddings

def compute_similarities(embeddings: np.ndarray, item_ids: List[str], k: int = 20) -> Dict:
    """Compute top-k similarities between embeddings"""
    print(f"Finding Top-{k} similar items...")
    
    n = len(embeddings)
    results = {}
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    start_time = time.time()
    
    # Compute similarities
    for i in tqdm(range(n), desc="Computing similarities"):
        similarities = np.dot(embeddings_norm[i], embeddings_norm.T)
        
        # Set self-similarity to -1
        similarities[i] = -1
        
        # Get top-k indices
        if k + 1 < n:
            top_k_indices = np.argpartition(similarities, -(k + 1))[-(k + 1):]
            top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
            top_k_indices = top_k_indices[1:k+1]
        else:
            top_k_indices = np.argsort(-similarities)[1:k+1]
        
        # Collect similar items
        similar_items = []
        for idx in top_k_indices:
            if idx != i:
                similar_items.append({
                    "item_id": item_ids[idx],
                    "similarity": float(similarities[idx])
                })
        
        results[item_ids[i]] = similar_items
    
    end_time = time.time()
    print(f"Similarity computation finished in {end_time - start_time:.2f}s")
    
    return results

def main(dataset):
    model_name = "../hf_qwen3_emb_8b"
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("Warning: No GPU available, will use CPU (slow)")
        num_gpus = 1
    
    data_file = f"./{dataset}/raw_data/{dataset}.item.json"
    print(f"Loading data: {data_file}")
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")
    
    # Check data
    if len(data) == 0:
        print("Error: No data loaded!")
        return
    
    # Generate embeddings using multiple GPUs
    results_with_embeddings, embeddings = generate_embeddings_multi_gpu(
        data, model_name, num_gpus=num_gpus, batch_size=16
    )
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings data type: {embeddings.dtype}")
    
    # Extract item IDs
    item_ids = [item['id'] for item in results_with_embeddings]
    
    # Compute similarities
    similarity_results = compute_similarities(embeddings, item_ids, k=20)
    
    # Save results
    output_file = f"./{dataset}/sum_data/{dataset}_similarities.json"
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, ensure_ascii=False, indent=2)
    
    
    print("\n" + "="*50)
    print("Processing completed!")
    print("="*50)

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Check PyTorch version and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    datasets = ["beauty", "sports", "toys"]
    for dataset in datasets:
        main(dataset)