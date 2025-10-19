src/lora.py contains code of LoRA, MoA, and merging strategies implementation. 

To run all the process in report, do

bash train_all_tasks.sh
bash train_moa.sh
bash run_all_evaluation.sh

To run evaluations on the trained weights, just do

bash run_all_evaluation.sh

for 7 tasks, 
change full_evaluation.py line 30 to tasks = ["summarization", "translation", "qa", "sentiment", "nli", "paraphrasing", "commonsense"]