import json
import numpy as np
from typing import List, Dict, Any

def hit_k(topk_results, k):
    """Calculate hit@k metric"""
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results)


def ndcg_k(topk_results, k):
    """Calculate ndcg@k metric"""
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Best case: hit at position 1
        ndcg += dcg / idcg
    return ndcg / len(topk_results)


def get_metrics_results(topk_results, metrics):
    """Calculate evaluation metrics"""
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    
    return res

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load inference results file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} sample results")
    return data

def create_topk_results(data: List[Dict[str, Any]], max_k: int = 200) -> List[List[int]]:
    """
    Create topk result matrix
    Each sample corresponds to a list indicating whether the ground truth item was hit at each topk position
    """
    topk_results = []
    
    for sample in data:
        # Get predicted item_id list and ground truth item_id
        predicted_iids = sample.get("iids", [])
        ground_truth_iid = sample.get("iid_gt", "")
        
        # Create hit sequence: 1 indicates hit, 0 indicates miss
        hit_sequence = []
        for i, pred_iid in enumerate(predicted_iids):
            if i >= max_k:
                break
            if pred_iid == ground_truth_iid:
                hit_sequence.append(1)
            else:
                hit_sequence.append(0)
        
        # If sequence length is less than max_k, pad with zeros
        while len(hit_sequence) < max_k:
            hit_sequence.append(0)
            
        topk_results.append(hit_sequence)
    
    print(f"Created topk results for {len(topk_results)} samples, up to {max_k} positions per sample")
    return topk_results

def analyze_hit_positions(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze hit position distribution"""
    hit_positions = []
    missed_samples = 0
    
    for sample in data:
        predicted_iids = sample.get("iids", [])
        ground_truth_iid = sample.get("iid_gt", "")
        
        hit_pos = -1
        for pos, pred_iid in enumerate(predicted_iids):
            if pred_iid == ground_truth_iid:
                hit_pos = pos + 1  # Convert to 1-based index
                break
        
        if hit_pos != -1:
            hit_positions.append(hit_pos)
        else:
            missed_samples += 1
    
    if hit_positions:
        stats = {
            "total_samples": len(data),
            "hit_samples": len(hit_positions),
            "missed_samples": missed_samples,
            "hit_rate": len(hit_positions) / len(data),
            "mean_hit_position": np.mean(hit_positions),
            "median_hit_position": np.median(hit_positions),
            "min_hit_position": np.min(hit_positions),
            "max_hit_position": np.max(hit_positions),
            "hit_at_1": sum(1 for pos in hit_positions if pos == 1),
            "hit_at_5": sum(1 for pos in hit_positions if pos <= 5),
            "hit_at_10": sum(1 for pos in hit_positions if pos <= 10),
        }
    else:
        stats = {
            "total_samples": len(data),
            "hit_samples": 0,
            "missed_samples": missed_samples,
            "hit_rate": 0.0
        }
    
    return stats

def main(dataset):
    # Configure file paths
    with open(f'./{dataset}/sum_data/item_id2tid/{dataset}_tid2item_id.json', 'r', encoding='utf-8') as f:
        tid2item_id = json.load(f)
    results_file = f"./{dataset}/rec_res/{dataset}_eval_info.json"
    
    # Define metrics to evaluate
    metrics = [
        "hit@1", "hit@3", "hit@5", "hit@10", "hit@20",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20"
    ]
    
    print("Starting evaluation of inference results...")
    
    # 1. Load result data
    data = load_results(results_file)
    
    # 2. Create topk result matrix
    topk_results = create_topk_results(data, max_k=20)
    
    # 3. Calculate evaluation metrics
    evaluation_results = get_metrics_results(topk_results, metrics)
    
    # 4. Analyze hit position distribution
    hit_stats = analyze_hit_positions(data)
    
    # 5. Merge all results
    final_results = {
        "evaluation_metrics": evaluation_results,
        "hit_statistics": hit_stats,
        "total_samples_evaluated": len(data)
    }
    
    # 6. Output results
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    print("\n=== Main Evaluation Metrics ===")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Hit Position Statistics ===")
    for stat, value in hit_stats.items():
        if isinstance(value, float):
            print(f"{stat}: {value:.4f}")
        else:
            print(f"{stat}: {value}")
    
    
    # 8. Output some sample-level analysis
    print("\n=== Sample-Level Analysis ===")
    print(f"Total samples: {len(data)}")
    print(f"Successful hits: {hit_stats['hit_samples']}")
    print(f"Missed samples: {hit_stats['missed_samples']}")
    print(f"Overall hit rate: {hit_stats['hit_rate']:.4f}")

if __name__ == "__main__":
    datasets = ["beauty", "sports", "toys"]
    for dataset in datasets:
        main(dataset)