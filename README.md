<h1 align='center'>Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers</h1>

<p align="center">
  <a href="[https://arxiv.org/abs/2601.06798](https://arxiv.org/abs/2601.06798)"><img src="https://img.shields.io/badge/Paper-ArXiv-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href="https://github.com/ZY0025/GRLM"><img src="https://img.shields.io/badge/GitHub-GRLM-black?logo=github" alt="GitHub"></a>
</p>

## Introduction

This is the official open-source repository for the paper **"Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers"**, which aims to build **LLM-based, general-purpose, and semantically-aware** recommendation systems. Our method has been applied to the text-augmented itemic tokens strategy of [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) and has achieved improved results.

Complete intermediate data and evaluation results can be obtained from: https://drive.google.com/file/d/1QLAMH0rNIRIGfCVWt6oBnPaCtgFY5JIU/view?usp=sharing.

## Environment Setup

The data processing, model fine-tuning, and evaluation in our experiments can be performed directly using the environment from **[LlamaFactory](https://github.com/hiyouga/LlamaFactory/)**. The setup process is as follows:

```bash
git clone https://github.com/ZY0025/GRLM.git
cd GRLM

conda create -n grlm python=3.11
conda activate grlm

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/deepspeed.txt
```

Additionally, download the model for fine-tuning, **[Qwen3-2507-4b](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**, and the model for retrieval, **[Qwen3-emb-8b](https://huggingface.co/Qwen/Qwen3-Embedding-8B)**, into the folders `hf_qwen3_2507_4b` and `hf_qwen3_emb_8b` respectively. If you already have the models or wish to use other models, you can replace the paths in the code with your own.

## Reproduction Steps

To facilitate understanding and reproduction, the main workflow of this repository is organized in the `in_domain` and `cross_domain` folders. You can adjust the `num_gpus` parameter in the code according to your resources. The following example assumes **8 A100 GPUs**.

### Data Processing

1. **Generate Item Similarities**
    ```
    python s0_init_emb.py
    ```
    Uses an embedding model to retrieve the top-k most similar items for each item.  
    **Input:** `{dataset}.item.json`  
    **Output:** `{dataset}_similarities.json`

2. **Extract Term IDs for Items**
    ```
    python s1_init_sum.py
    ```
    Uses an LLM to extract Term IDs for each item.  
    **Input:** `{dataset}.item.json` and `{dataset}_similarities.json`  
    **Output:** `{dataset}_summaries_with_similarity.jsonl`  
    **Note:** LLM generation is stochastic, so regular expressions may not always extract exactly five Term IDs. This step will output a legality rate. You can manually adjust the invalid ones. If no adjustments are made, subsequent steps will skip invalid items by default.

3. **Build ID-to-Metadata Mapping**
    ```
    python s2_build_id2meta.py
    ```
    Constructs a JSON mapping from item ID to item metadata.  
    **Input:** `{dataset}_summaries_with_similarity.jsonl`  
    **Output:** `{dataset}_id2meta.json`

4. **Build SFT Data for Generative Term Internalization**
    ```
    python s3_build_meta2tid_sft_data.py
    ```
    Constructs SFT data for the Generative Term Internalization task. The data generated in this and the next two steps will be saved directly to the `data` folder of LlamaFactory.

5. **Build SFT Data for User Behavior Sequence Prediction**
    ```
    python s4_build_rec_sft_data.py
    ```
    Constructs SFT data for the User Behavior Sequence Prediction task.

6. **Build SFT Data for Collaborative Signals (New Optimization)**
    ```
    python s4_build_collaborative_sft_data.py
    ```
    This is a **new optimization not included in the original paper**. It explicitly incorporates collaborative signals between items (modeled via SASRec) as training data, effectively enhancing model performance. For example, on the Beauty dataset, Recall@5 increased from **0.0607 to 0.0664**.

### Model Training

We use the popular **LlamaFactory** framework for model fine-tuning.

Navigate to `LlamaFactory/data/grlm_in_domain` and run the `merge_dataset.py` file (you can incorporate the new collaborative training data from `s4_build_collaborative_sft_data.py` here). Then, go to the `LlamaFactory` directory and run the training script, such as `grlm_indomain_beauty.sh`, to start training.

For improved performance, consider adding the following hyperparameters to your training script:
```
--optim adamw_torch \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--warmup_ratio 0.1
```

### Model Evaluation

Return to the `GRLM/in_domain` folder and run `s5_beauty_eval.py` to begin model evaluation. This script logs the model's raw outputs, maps them to items, and calculates the recall metric. You can then use `s6_post_eval.py` to obtain the full Recall and NDCG metrics.

## Final Notes

- **TIDs Collision:** Thanks to the [issue](https://github.com/ZY0025/GRLM/issues/1) raised by Luo-Jiaming, Term IDs can also experience collisions. However, the collision rate remains relatively low (around 2-3% in our datasets). As dataset size increases, the TIDs collision rate tends to stay within this range.
- **Distribution of TIDs:** An interesting phenomenon occurs during Term ID generation. For example, in the Beauty dataset, using an LLM to extract Term IDs directly yields around **4,000 unique IDs**. However, when context-aware extraction is introduced, this number rises to approximately **7,500**, and performance improves. This contrasts with traditional statistical recommendation systems, where higher frequency of similar patterns is typically favored.

As a basic work, **GRLM's potential is far from fully realized**. Improvements in Term ID generation, optimization of fine-tuning tasks, or even simple adjustments to training scripts can lead to noticeable gains in performance metrics. Our codebase includes several optimizations introduced after the paper's publication, such as incorporating explicit collaborative signals as a new training task and refining the training scripts. We are considering to include these in an updated version of the paper.

If you find our work helpful, we welcome you to give us a ⭐ or cite our paper. If you encounter any difficulties during reproduction or are interested in more technical details, please feel free to open an issue or contact us via email at `zhangzhiyang06@kuaishou.com` or `415573678@qq.com`.

```bibtex
@article{zhang2026unleashing,
  title={Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers},
  author={Zhang, Zhiyang and She, Junda and Cai, Kuo and Chen, Bo and Wang, Shiyao and Luo, Xinchen and Luo, Qiang and Tang, Ruiming and Li, Han and Gai, Kun and others},
  journal={arXiv preprint arXiv:2601.06798},
  year={2026}
}
```
