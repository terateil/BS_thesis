# Rethinking LoRA Merging for Language Model’s Multitask Adaptation

Author: Chanbin Lee  
Affiliation: UNIST, Department of Computer Science and Engineering  
Email: chblee@unist.ac.kr

------------------------------------------------------------

## Overview

This repository implements and evaluates LoRA (Low-Rank Adaptation), Mixture-of-LoRAs (MoA), and several routing-free merging strategies for multitask adaptation of large language models.

The key idea explored in this work is:
Can we achieve multitask adaptation without explicit routing, simply by merging LoRA modules when input prompts are well-structured?

Our experiments show that simple merging strategies (concatenation, weight summing, SVD-based compression) can achieve performance comparable to MoA, especially in large models or when tasks are clearly distinguishable through structured prompts.

------------------------------------------------------------


## Project Structure

```
Merging_Loras/
│
├── src/
│   └── lora.py                # Core implementation of LoRA, MoA, and merging strategies
│
├── results/                   # Evaluation results and logs
│
├── train_all_tasks.sh         # Run training for all tasks
├── train_moa.sh               # Run MoA (Mixture of LoRAs) training
├── run_all_evaluation.sh      # Evaluate all trained models
├── full_evaluation.py         # Unified evaluation script across tasks
├── train_individual_loras.py  # Train LoRAs separately for each task
├── train_moa.py               # Implementation of MoA training
├── visualize_lora.ipynb       # Notebook for visualization and analysis
│
├── Thesis.pdf                 # Full research paper
└── README.md                  # This file
```

------------------------------------------------------------

## How to Run

1. Train LoRA modules for all tasks:
bash train_all_tasks.sh

2. Train Mixture-of-LoRAs (MoA):
bash train_moa.sh

3. Run Evaluation:
bash run_all_evaluation.sh

If you already have trained weights, you can directly run the above command — training is not required again.

------------------------------------------------------------

## Supported Tasks

The project supports 7 NLP tasks:

1. Summarization
2. Translation
3. Question Answering (QA)
4. Sentiment Analysis
5. Natural Language Inference (NLI)
6. Paraphrasing
7. Commonsense Reasoning

To activate all tasks, modify line 30 of full_evaluation.py:

tasks = ["summarization", "translation", "qa", "sentiment", "nli", "paraphrasing", "commonsense"]

------------------------------------------------------------

## Implementation Details

1. LoRA Training
Each task has its own LoRA module, fine-tuned independently with structured task-specific prompts (e.g., QA format includes Context, Question, Answer).
LoRA is inserted into attention projection layers (rank = 8).

2. Mixture-of-LoRAs (MoA)
- Introduces a router to select task-specific LoRA modules during inference.
- Router is trained as a sequence-level classifier over hidden states.
- Achieves near-oracle multitask performance by accurately routing based on structured prompts.

3. Routing-Free Merging
Implemented strategies in src/lora.py:
- Concatenation: Apply all LoRA modules in parallel, sum their outputs.
- Weight Summing: Linearly combine LoRA weights with scalar coefficients.
- SVD Merge: Average and compress LoRA weights via singular value decomposition.

These methods eliminate routing overhead while maintaining strong multitask capability.

------------------------------------------------------------

## Experimental Setup

Base Model: Qwen2 (0.5B and 1.5B parameters)
LoRA Rank: 8
Frameworks: PyTorch, Transformers
Datasets: CNN/DailyMail, WMT14, SQuAD v1.1, SST-2, MNLI, PAWS, PIQA
Evaluation Metrics:
- ROUGE (Summarization)
- BLEU (Translation / Paraphrasing)
- EM / F1 (QA)
- Accuracy (Classification tasks)

------------------------------------------------------------

## Results Summary

Model                | 5-task (1.5B) | 7-task (1.5B)
----------------------|----------------|----------------
Base                 | 0.224          | 0.242
MoA                  | 1.004          | 1.002
SVD Merge            | 0.731          | 0.704
Weight Summing       | 0.956          | 0.790
Concatenation        | 0.955          | 0.771

Key Findings:
- MoA performs near-oracle across all tasks.
- Concatenation and Weight Summing perform surprisingly well at large scales.
- SVD Merge offers compact, balanced performance.
- Router-free methods degrade as the number of tasks increases.

------------------------------------------------------------
