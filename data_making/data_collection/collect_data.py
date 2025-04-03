from datasets import Dataset, load_dataset, concatenate_datasets
import pathlib
from load_test_datasets import load_test_datasets
import re
import pandas as pd
from utils import file_exists, decontaminate_train_data, self_decontaminate_train_data, filter_TFIDF_similar_strings, self_filter_TFIDF_similar_strings, self_sentence_embedding_similarity, sentence_embedding_similarity
import random
import csv
import json

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"
# DS_COLUMNS = ["question", "source", "id", "category"]
DS_COLUMNS = ["question", "source"]
    
def load_SaladData() -> Dataset:
    if file_exists(str(DATA_DIR)+"/SaladData/SaladData.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/SaladData/SaladData.csv"))
    ds = load_dataset(
        "OpenSafetyLab/Salad-Data",
        name="base_set",
        cache_dir=str(DATA_DIR)+"/SaladData",
    )['train']
    ds = ds.map(lambda x: {"source": "SaladBench", "id": x.pop("qid"), "category": f"First-level: {x.pop('1-category')}; Second-level: {x.pop('2-category')}; Third-level: {x.pop('3-category')}", "question": x.pop("question")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/SaladData/SaladData.csv", index=False)
    return ds

def load_DoNotAnswer() -> Dataset:
    if file_exists(str(DATA_DIR)+"/DoNotAnswer/DoNotAnswer.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/DoNotAnswer/DoNotAnswer.csv"))
    ds = load_dataset(
        "LibrAI/do-not-answer",
        cache_dir=str(DATA_DIR)+"/DoNotAnswer",
    )['train']
    ds = ds.map(lambda x: {"source": "DoNotAnswer", "id": x.pop("id"), "category": x.pop("risk_area")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/DoNotAnswer/DoNotAnswer.csv", index=False)
    return ds

def load_wildjailbreak() -> Dataset:
    if file_exists(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"))
    ds = load_dataset(
        "allenai/wildjailbreak",
        name="train",
        delimiter="\t", 
        keep_default_na=False,
        cache_dir=str(DATA_DIR)+"/wildjailbreak"
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("vanilla"), "source": "wildjailbreak"})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv", index=False)
    return ds

def load_UltraSafety() -> Dataset:
    if file_exists(str(DATA_DIR)+"/UltraSafety/UltraSafety.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/UltraSafety/UltraSafety.csv"))
    ds = load_dataset(
        "openbmb/UltraSafety",
        cache_dir=str(DATA_DIR)+"/UltraSafety"
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("instruction"), "source": "UltraSafety"})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/UltraSafety/UltraSafety.csv", index=False)
    return ds


def load_GPTFuzz() -> Dataset:
    if file_exists(str(DATA_DIR)+"/GPTFuzz/GPTFuzz.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/GPTFuzz/GPTFuzz.csv"))
    ds = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/sherdencooper/GPTFuzz/refs/heads/master/datasets/questions/question_list.csv",
        cache_dir=str(DATA_DIR)+"/GPTFuzz",
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("text"), "source": "GPTFuzz", "id": x.pop("index")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/GPTFuzz/GPTFuzz.csv", index=False)
    return ds

def load_ALERT() -> Dataset:
    if file_exists(str(DATA_DIR)+"/ALERT/ALERT.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/ALERT/ALERT.csv"))
    ds = load_dataset(
        "Babelscape/ALERT",
        cache_dir=str(DATA_DIR)+"/ALERT",
        name="alert",
    )['test']
    ds = ds.map(lambda x: {"question": re.sub(r'### (Instruction:|Response:)\n?', '', x.pop("prompt")).strip(), "source": "ALERT", "category": x.pop("category"), "id": x.pop("id")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/ALERT/ALERT.csv", index=False)
    return ds

def load_HarmBench() -> Dataset:
    if file_exists(str(DATA_DIR)+"/HarmBench/HarmBench.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/HarmBench/HarmBench.csv"))
    ds = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        cache_dir=str(DATA_DIR)+"/HarmBench",
    )['train']
    ds = ds.filter(lambda x: x["FunctionalCategory"] != "contextual")
    ds = ds.map(lambda x: {"question": x.pop("Behavior"), "source": "HarmBench", "category": x.pop("SemanticCategory")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/HarmBench/HarmBench.csv", index=False)
    return ds

def load_SimpleSafetyTests() -> Dataset:
    if file_exists(str(DATA_DIR)+"/SimpleSafetyTests/SimpleSafetyTests.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/SimpleSafetyTests/SimpleSafetyTests.csv"))
    ds = load_dataset(
        "Bertievidgen/SimpleSafetyTests",
        cache_dir=str(DATA_DIR)+"/SimpleSafetyTests",
    )['test']
    ds = ds.map(lambda x: {"question": x.pop("prompt"), "source": "SimpleSafetyTests", "category": x.pop("harm_area"), "id": int(x.pop("id").split('_')[-1])})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/SimpleSafetyTests/SimpleSafetyTests.csv", index=False)
    return ds

def load_MaliciousInstruct() -> Dataset:
    if file_exists(str(DATA_DIR)+"/MaliciousInstruct/MaliciousInstruct.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/MaliciousInstruct/MaliciousInstruct.csv"))
    ds = load_dataset(
        "text",
        data_files="https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/refs/heads/main/data/MaliciousInstruct.txt",
        cache_dir=str(DATA_DIR)+"/MaliciousInstruct",
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("text"), "source": "MaliciousInstruct"})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/MaliciousInstruct/MaliciousInstruct.csv", index=False)
    return ds


def load_HExPHI() -> Dataset:
    if file_exists(str(DATA_DIR)+"/HExPHI/HExPHI.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/HExPHI/HExPHI.csv"))
    ds = load_dataset(
        "LLM-Tuning-Safety/HEx-PHI",
        cache_dir=str(DATA_DIR)+"/HExPHI",
        streaming=True,
    )  
    ds_all = []
    for split in ds:
        ds_split = ds[split]
        values = []
        for x in ds_split:
            values.append(list(x.values())[0])         
        values.append(list(x.keys())[0])
        ds_split = Dataset.from_list([{"question": x, "source": "HExPHI", "category": split} for x in values])
        ds_all.append(ds_split)
    ds = concatenate_datasets(ds_all)
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/HExPHI/HExPHI.csv", index=False)
    return ds

def load_QHarm() -> Dataset:
    if file_exists(str(DATA_DIR)+"/QHarm/QHarm.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/QHarm/QHarm.csv"))
    ds = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/data/evaluation/Q-Harm.json",
        cache_dir=str(DATA_DIR)+"/QHarm",
    )['train']
    ds = Dataset.from_list([{"question": x, "source": "QHarm"} for x in ds["instructions"][0]])
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/QHarm/QHarm.csv", index=False)
    return ds

def load_MaliciousInstructions() -> Dataset:
    if file_exists(str(DATA_DIR)+"/MaliciousInstructions/MaliciousInstructions.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/MaliciousInstructions/MaliciousInstructions.csv"))
    ds = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/data/evaluation/I-MaliciousInstructions.json",
        cache_dir=str(DATA_DIR)+"/MaliciousInstructions",
    )['train']
    ds = Dataset.from_list([{"question": x, "source": "MaliciousInstructions", "category": y} for x, y in zip(ds["instructions"][0], ds["tags"][0])])
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/MaliciousInstructions/MaliciousInstructions.csv", index=False)
    return ds

def load_SafetyInstructions() -> Dataset:
    if file_exists(str(DATA_DIR)+"/SafetyInstructions/SafetyInstructions.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/SafetyInstructions/SafetyInstructions.csv"))
    ds = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/data/training/safety_only_data_Instructions.json",
        cache_dir=str(DATA_DIR)+"/SafetyInstructions",
    )['train']
    ds = ds.map(lambda x: {"source": "SafetyInstructions", "question": x.pop("instruction")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/SafetyInstructions/SafetyInstructions.csv", index=False)
    return ds


def load_BeaverTails() -> Dataset:
    if file_exists(str(DATA_DIR)+"/BeaverTails/BeaverTails.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/BeaverTails/BeaverTails.csv"))
    ds_train = load_dataset(
        "PKU-Alignment/BeaverTails",
        cache_dir=str(DATA_DIR)+"/BeaverTails",
    )['330k_train']
    ds_test = load_dataset(
        "PKU-Alignment/BeaverTails",
        cache_dir=str(DATA_DIR)+"/BeaverTails",
    )['330k_test']
    ds = concatenate_datasets([ds_train, ds_test])
    ds = ds.filter(lambda x: x["is_safe"] == False)
    
    def transform_example(x):
        # Convert string to dictionary
        category_dict = x.pop("category")
        
        # Extract keys with True values
        true_keys = [key for key, value in category_dict.items() if value]
        
        # Select a random key if there are multiple, otherwise None
        x["category"] = random.choice(true_keys) if true_keys else None
        
        # Rename prompt column and add source
        x["question"] = x.pop("prompt")
        x["source"] = "BeaverTails"
        return x
    
    ds = ds.map(transform_example)
    
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/BeaverTails/BeaverTails.csv", index=False)
    return ds

def load_PKUSafeRLHF() -> Dataset:
    if file_exists(str(DATA_DIR)+"/PKUSafeRLHF/PKUSafeRLHF.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/PKUSafeRLHF/PKUSafeRLHF.csv"))
    ds_train = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF",
        cache_dir=str(DATA_DIR)+"/PKUSafeRLHF",\
        name="default",
    )['train']
    ds_test = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF",
        cache_dir=str(DATA_DIR)+"/PKUSafeRLHF",
        name="default",
    )['test']
    ds = concatenate_datasets([ds_train, ds_test])
    ds = ds.filter(lambda x: x["is_response_0_safe"] == False and x["is_response_1_safe"] == False)
    def merge_harm_categories(x):
        harm_0 = x.pop("response_0_harm_category", {})
        harm_1 = x.pop("response_1_harm_category", {})
        
        merged_harm = {key: harm_0.get(key, False) and harm_1.get(key, False) for key in set(harm_0) | set(harm_1)}
        
        return {"response_0_harm_category": merged_harm}
    ds = ds.map(merge_harm_categories)
    
    def transform_example(x):
        # Convert string to dictionary
        category_dict = x.pop("response_0_harm_category")
        
        # Extract keys with True values
        true_keys = [key for key, value in category_dict.items() if value]
        
        # Select a random key if there are multiple, otherwise None
        x["category"] = random.choice(true_keys) if true_keys else None
        
        # Rename prompt column and add source
        x["question"] = x.pop("prompt")
        x["source"] = "PKUSafeRLHF"
        return x

    ds = ds.map(transform_example)
    
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/PKUSafeRLHF/PKUSafeRLHF.csv", index=False)
    return ds

def load_TDCRedTeaming() -> Dataset:
    if file_exists(str(DATA_DIR)+"/TDCRedTeaming/TDCRedTeaming.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/TDCRedTeaming/TDCRedTeaming.csv"))
    ds_dev = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/refs/heads/main/red_teaming/data/dev/behaviors.json",
        cache_dir=str(DATA_DIR)+"/TDCRedTeaming",
    )['train']
    ds_test = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/refs/heads/main/red_teaming/data/test/behaviors.json",
        cache_dir=str(DATA_DIR)+"/TDCRedTeaming",
    )['train']
    ds = concatenate_datasets([ds_dev, ds_test])
    ds = ds.map(lambda x: {"source": "TDCRedTeaming", "question": x.pop("text")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/TDCRedTeaming/TDCRedTeaming.csv", index=False)
    return ds

def load_HarmfulQA() -> Dataset:
    if file_exists(str(DATA_DIR)+"/HarmfulQA/HarmfulQA.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/HarmfulQA/HarmfulQA.csv"))
    ds = load_dataset(
        "declare-lab/HarmfulQA",
        cache_dir=str(DATA_DIR)+"/HarmfulQA",
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("question"), "id": x.pop("id"), "source": "HarmfulQA"})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/HarmfulQA/HarmfulQA.csv", index=False)
    return ds

def load_AttaQ() -> Dataset:
    if file_exists(str(DATA_DIR)+"/AttaQ/AttaQ.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/AttaQ/AttaQ.csv"))
    ds = load_dataset(
        "ibm-research/AttaQ",
        cache_dir=str(DATA_DIR)+"/AttaQ",
    )['train']
    ds = ds.map(lambda x: {"question": x.pop("input"), "source": "AttaQ", "category": x.pop("label")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/AttaQ/AttaQ.csv", index=False)
    return ds

def load_HarmfulQ() -> Dataset:
    if file_exists(str(DATA_DIR)+"/HarmfulQ/HarmfulQ.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/HarmfulQ/HarmfulQ.csv"))
    ds = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/SALT-NLP/chain-of-thought-bias/refs/heads/main/data/dangerous-q/toxic_outs.json",
        cache_dir=str(DATA_DIR)+"/HarmfulQ",
    )['train']
    ds = ds.map(lambda x: {"source": "HarmfulQ", "question": x.pop("text")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/HarmfulQ/HarmfulQ.csv", index=False)
    return ds

TRAIN_DATASETS = {
    # Name: [load function, n-gram size]
    
    # 100 samples
    "GPTFuzz": [load_GPTFuzz, 8],  
    # 100 samples
    "SimpleSafetyTests": [load_SimpleSafetyTests, 8],
    # 100 samples
    "MaliciousInstruct": [load_MaliciousInstruct, 8],
    # 100 samples
    "QHarm": [load_QHarm, 8],
    # 100 samples
    "TDCRedTeaming": [load_TDCRedTeaming, 8],
    # 100 samples
    "MaliciousInstructions": [load_MaliciousInstructions, 8],
    # 200 samples
    "HarmfulQ": [load_HarmfulQ, 8],
    # 300 samples
    "HExPHI": [load_HExPHI, 8],
    # 300 samples (filter the contextual FunctionalCategory)
    "HarmBench": [load_HarmBench, 8],
    # 1.4k samples
    "AttaQ": [load_AttaQ, 7],
    # 2k samples
    "HarmfulQA": [load_HarmfulQA, 7],
    # 2.5k samples
    "SafetyInstructions": [load_SafetyInstructions, 7],
    # 3k samples
    "UltraSafety": [load_UltraSafety, 7],
    # 14.8k samples
    "ALERT": [load_ALERT, 7],
    # 21.3K samples (High-Quility)
    "SaladBench": [load_SaladData, 8],
    # 36.3k samples (filter the is_response_0_safe=True and is_response_1_safe=True)
    "PKUSafeRLHF": [load_PKUSafeRLHF, 6],
    # 185.1k samples (filter the is_safe=True)
    "BeaverTails": [load_BeaverTails, 5],
    # 261k samples
    "wildjailbreak": [load_wildjailbreak, 5],
}

def main():
    step1_file_tag = "step1_02211113"
    step2_file_tag = "step2_02211113"
    step3_file_tag = "step3_02211113"
    step4_file_tag = "step4_02211113"
    sentence_embedding_filter = True
    if file_exists(str(DATA_DIR)+f"/train_datasets_{step1_file_tag}.csv"):
        print("Loading step1 train datasets...")
        ds_all = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+f"/train_datasets_{step1_file_tag}.csv"))
        print("Step1 dataset:", ds_all, flush=True)
    else:
        test_datasets = load_test_datasets()
        test_questions = test_datasets['question']
        ds_all = []
        for dataset in TRAIN_DATASETS:
            print(f"Loading {dataset}...")
            ds = TRAIN_DATASETS[dataset][0]()
            ds = self_decontaminate_train_data(ds['question'], ds, ngram_size=TRAIN_DATASETS[dataset][1])
            ds = decontaminate_train_data(ds['question'], test_questions, ds, ngram_size=8)
            ds = filter_TFIDF_similar_strings(ds['question'], test_questions, ds, threshold=0.6)
            test_questions = test_questions + ds['question']
            ds_all.append(ds)
            print("----------------------------------------------------------")
        ds_all = concatenate_datasets(ds_all)
        df = ds_all.to_pandas()
        df.to_csv(str(DATA_DIR)+f"/train_datasets_{step1_file_tag}.csv", index=False)
        print("Step1 dataset:", ds_all)
        print("Step1 Done.", flush=True)
        
    if file_exists(str(DATA_DIR)+f"/train_datasets_{step2_file_tag}.csv"):
        print("Loading step2 train datasets...")
        ds = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+f"/train_datasets_{step2_file_tag}.csv"))
        print("Step2 dataset:", ds, flush=True)
    else:
        threshold = 0.6
        print("Self Filtering TFIDF similar strings...")
        print("Threshold: ", threshold, flush=True)
        ds = self_filter_TFIDF_similar_strings(ds_all['question'], ds_all, threshold=threshold, cache_file=f".cache/sim_matrix_{step1_file_tag}.npy")
        df = ds.to_pandas()
        df.to_csv(str(DATA_DIR)+f"/train_datasets_{step2_file_tag}.csv", index=False)
        print("Step2 dataset:", ds)
        print("Step2 Done.")
    
    if sentence_embedding_filter:
        if file_exists(str(DATA_DIR)+f"/train_datasets_{step3_file_tag}.csv"):
            print("Loading step3 train datasets...")
            ds = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+f"/train_datasets_{step3_file_tag}.csv"))
            print("Step3 dataset:", ds, flush=True)
        else:
            threshold = 0.7
            print("Self Filtering Embedding similar strings...")
            print("Threshold: ", threshold, flush=True)
            ds = self_sentence_embedding_similarity(ds['question'], ds, threshold=threshold, model_name="all-MiniLM-L6-v2", cache_file=f".cache/embed_sim_matrix_{step2_file_tag}.npy")
            df = ds.to_pandas()
            df.to_csv(str(DATA_DIR)+f"/train_datasets_{step3_file_tag}.csv", index=False)
            print("Step3 dataset:", ds)
            print("Step3 Done.")
        
        if file_exists(str(DATA_DIR)+f"/train_datasets_{step4_file_tag}.csv"):
            print("Loading step4 train datasets...")
            ds = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+f"/train_datasets_{step4_file_tag}.csv"))
            print("Step4 dataset:", ds, flush=True)
        else:
            threshold = 0.7
            print("Filtering Embedding similar strings...")
            print("Threshold: ", threshold, flush=True)
            test_datasets = load_test_datasets()
            ds = sentence_embedding_similarity(ds['question'], test_datasets['question'], ds, threshold=threshold, model_name="all-MiniLM-L6-v2")
            df = ds.to_pandas()
            df.to_csv(str(DATA_DIR)+f"/train_datasets_{step4_file_tag}.csv", index=False)
            print("Step4 dataset:", ds)
            print("Step4 Done.")
    
    df = pd.read_csv(str(DATA_DIR)+f"/train_datasets_{step4_file_tag}.csv")
    df['id'] = df.index
    
    csv_file = "data_collection.csv"
    json_file = "data_collection.json"
    
    df.to_csv(csv_file, index=False)
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()