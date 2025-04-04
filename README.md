# 🌟 STAR-1: Safer Alignment of Reasoning LLMs with 1K Data

<p align="center">
📃 <a href="https://arxiv.org/abs/2504.01903" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/datasets/UCSC-VLAA/STAR-1" target="_blank">STAR-1 Data</a> | 🤗 <a href="https://huggingface.co/collections/UCSC-VLAA/star-1-67edda2a042e8ba3e955e522" target="_blank">STAR-1 Model</a> |  📚 <a href="https://ucsc-vlaa.github.io/STAR-1/" target="_blank">Project Page</a>
</p>

[Zijun Wang](https://asillycat.github.io/), [Haoqin Tu](https://www.haqtu.me/), [Yuhan Wang](https://scholar.google.com/citations?user=Bo9xeqMAAAAJ&hl=en), [Juncheng Wu](https://chtholly17.github.io/), [Jieru Mei](https://meijieru.com/), [Brian R. Bartoldson](https://brianbartoldson.wordpress.com/), [Bhavya Kailkhura](https://people.llnl.gov/kailkhura1), [Cihang Xie](https://cihangxie.github.io/)

## Introduction

<img src="./assets/SART1_teaser_final.jpg" alt="main" style="zoom: 33%;" />

[**STAR-1**](https://huggingface.co/datasets/UCSC-VLAA/STAR-1) is a high-quality safety dataset designed to enhance safety alignment in large reasoning models (LRMs) like DeepSeek-R1.

- Built on the principles of diversity, deliberative reasoning, and rigorous filtering, STAR-1 integrates and refines data from multiple sources to provide policy-grounded reasoning samples.
- The dataset contains **1,000** carefully selected examples, each aligned with best safety practices through GPT-4o-based evaluation.
- Fine-tuning with STAR-1 leads to significant safety improvements across multiple benchmarks, with minimal impact on reasoning capabilities.


## Artifacts
### Data

| Dataset    | Num. of Sample | URL                                                                 |
|------------|----------------|----------------------------------------------------------------------|
| STAR-1     | 1K             | 🤗 [UCSC-VLAA/STAR-1](https://huggingface.co/datasets/UCSC-VLAA/STAR-1) |
| STAR 41K   | 41K            | 🤗 [UCSC-VLAA/STAR-41K](https://huggingface.co/datasets/UCSC-VLAA/STAR-41K) |
| STAR-benign-915   | 915            | 🤗 [UCSC-VLAA/STAR-benign-915](https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915) |



### Model
| Model                          | Type                                      | URL                                                                                   |
|--------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------|
| `STAR1`-R1-Distill-1.5B        | R1-Distill-Qwen-1.5B trained on STAR-1    | 🤗 [UCSC-VLAA/STAR1-R1-Distill-1.5B](https://huggingface.co/UCSC-VLAA/STAR1-R1-Distill-1.5B) |
| `STAR1`-R1-Distill-7B          | R1-Distill-Qwen-7B trained on STAR-1      | 🤗 [UCSC-VLAA/STAR1-R1-Distill-7B](https://huggingface.co/UCSC-VLAA/STAR1-R1-Distill-7B)     |
| `STAR1`-R1-Distill-8B          | R1-Distill-Llama-8B trained on STAR-1     | 🤗 [UCSC-VLAA/STAR1-R1-Distill-8B](https://huggingface.co/UCSC-VLAA/STAR1-R1-Distill-8B)     |
| `STAR1`-R1-Distill-14B         | R1-Distill-Qwen-14B trained on STAR-1     | 🤗 [UCSC-VLAA/STAR1-R1-Distill-14B](https://huggingface.co/UCSC-VLAA/STAR1-R1-Distill-14B)   |
| `STAR1`-R1-Distill-32B         | R1-Distill-Qwen-32B trained on STAR-1     | 🤗 [UCSC-VLAA/STAR1-R1-Distill-32B](https://huggingface.co/UCSC-VLAA/STAR1-R1-Distill-32B)   |


## Structure
- `data_making/`: STAR-1 Data making pipeline (Sec. 2)

    - `data_collection/`: Sec. 2.1
    - `deliberative_reasoning/`: Sec. 2.2
    -  `data_selection`: Sec. 2.3
- `train/`: Training scripts (Sec. 3.1)
- `benchmark/`: Evaluation Scripts (Sec. 3.1)

    - `safe_benchmark`: Safety Evaluation 
    - `reasoning_benchmark/`: Reasoning Evaluation
- `overrefusal_ablation/`: A Mitigation for the Overrefusal Behaviour (Sec. 4.3)

## Quick Start
```
git clone https://github.com/UCSC-VLAA/STAR-1.git
cd STAR-1
pip install -e .
```

## Data Making Pipeline (Sec 2)
### Data Collection (Sec. 2.1)
```
cd data_making/data_collection/scripts
bash load_test_datasets.sh
bash collect_decon_train_datasets.sh
```
You will get a json file named `data_collection.json` under `data_making/data_collection/` folder and this contains the 41K initial collected data (after deduplication).

### Deliberative Reasoning (Sec. 2.2)
```
cd data_making/deliberative_reasoning/category_classification
mkdir datasets
cp ../../data_collection/data_collection.json datasets/
bash category_classification.sh
```
You will get a json file named `data_with_category.json` under `data_making/deliberative_reasoning/category_classification/` folder and this contains the 41K samples with safety categories classified by GPT-4o.

```
cd data_making/deliberative_reasoning/reasoning_generation
mkdir datasets
cp ../category_classification/data_with_category.json datasets/
bash reasoning_generation.sh
```
You will get a json file named `data_with_cot.json` under `data_making/deliberative_reasoning/reasoning_generation/` folder and this contains the 41K samples with safety-aligned reasoning process grounded with safety policies, generated by Deepseek-R1.

### Data Selection (Sec. 2.3)
```
cd data_making/data_selection
mkdir datasets
cp ../deliberative_reasoning/reasoning_generation/data_with_cot.json datasets/
bash scorer.sh
```
You will get:
- A json file named `data_with_score.json` under `data_making/data_selection/` folder and this contains the 41K samples with 4o-based scores. This `data_with_score.json` is our [`
STAR-41K`](https://huggingface.co/datasets/UCSC-VLAA/STAR-41K)
- A json file named `all_10.json` under `data_making/data_selection/` folder and this contains samples with full scores(10) on all criteria. (Sec. 2.3 - `Ensuring Accuracy`)
- A json file named `star1_high.json` under `data_making/data_selection/` folder and this contains 1K samples selected according to balanced representation. (Sec. 2.3 - `Ensuring Diversity`). This `star1_high.json` is our final [`STAR-1`](https://huggingface.co/datasets/UCSC-VLAA/STAR-1).

## Training (Sec 3.1)
```
cd train
bash run_sft.sh
```
The `run_sft.sh ` looks like:
```
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --train_bsz_per_gpu 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ../data/STAR-1.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1
```
- `base_flag`: If distill model then 0 elif instruct model then 1
- `think_flag`: default=1, if `w/o think` then 0 **(Sec 4.2)**
- `train_bsz_per_gpu * num_processes` should be 8 to keep the batchsize as 128
- You change the `model_path` to different model 
- Or change the `data_path` to use different finetune data **(Sec 4.1)**

## Evaluation (Sec 3.1)
You could change the `mode_path` of the evaluated model in `benchmark/config.py`.
### Safety Benchmark
```
cd benchmark/safe_benchmark
bash scripts.sh $model $data
# bash scripts.sh DeepSeek-R1-Distill-Qwen-1.5B strongreject
```

### Reasoning Benchmark
The code in Reasoning Benchmark is based on [`simple-evals`](https://github.com/openai/simple-evals) and modified.
```
cd benchmark/reasoning_benchmark
bash run_all_evals.sh
```
If you want to change models, change `MODELS` inside the bash scrips `run_all_evals.sh` at Line 7.

## Overrefusal Ablation (Sec 4.3)
### Data Generation (App. E)
```
cd overrefusal_ablation/benign_gene
mkdir datasets
cp ../../data_making/data_selection/star1_high.json datastes/
bash rewriter.sh
```
You will get a json file named `star1_benign.json` under `overrefusal_ablation/benign_gene/` folder and this contains the 1K samples that structurally similar to 1K harmful questions in STAR-1 but benign variants.


### Reasoning Generation
```
cd overrefusal_ablation/reasoning_generation
mkdir datasets
cp ../benign_gene/star1_benign.json datasets/
bash reasoning_generation.sh
```
You will get a json file named `star1_benign_with_cot.json` under `overrefusal_ablation/reasoning_generation/` folder and this contains 1K benign variants with reasoning process generated by Deepseek-R1.

### Data Filtering (App. E)
```
cd overrefusal_ablation/scorer
mkdir datasets
cp ../reasoning_generation/star1_benign_with_cot.json datasets/
bash scorer.sh
```
You will get 
- A json file named `star1_benign_with_score.json` under `overrefusal_ablation/scorer/` folder and this contains the 1K benign variants with 4o-based scores.
- A json file named `star1_benign_filtered.json` under `overrefusal_ablation/scorer/` folder and this contains the benign variants with full scores(5) on all criteria. This `star1_benign_filtered.json` is our final [`
STAR-benign-915 `](https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915).

### Training and Benchmarking
Then you can combine the `star1_benign_filtered.json` with `star1_high.json`, and use these 2K samples to sft a model and benchmark the finetuned model. The pipeline is the same as `Training` and `Evaluation` above.


## Acknowledgement
This work is partially supported by a gift from Open Philanthropy. We thank the NAIRR Pilot Program and the Microsoft Accelerate Foundation Models Research Program for supporting our computing needs.

LLNL co-authors were supported under Contract DE-AC52-07NA27344 with the U.S. Department of Energy and the LLNL-LDRD Program under Project No. 24-ERD-058. The United States Government retains, and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes.


## Citation
```
@article{wang2025star1saferalignmentreasoning,
    title={STAR-1: Safer Alignment of Reasoning LLMs with 1K Data}, 
    author={Zijun Wang and Haoqin Tu and Yuhan Wang and Juncheng Wu and Jieru Mei and Brian R. Bartoldson and Bhavya Kailkhura and Cihang Xie},
    year={2025},
    journal = {arXiv preprint arXiv:2504.01903}
}
```




