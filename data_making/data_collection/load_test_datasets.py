from datasets import Dataset, load_dataset, concatenate_datasets
import pathlib
import re
from utils import file_exists
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "datasets"
DS_COLUMNS = ["question", "source", "id", "category", "category_score"]

def load_StrongREJECT() -> Dataset:
    if file_exists(str(DATA_DIR)+"/StrongREJECT/StrongREJECT.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/StrongREJECT/StrongREJECT.csv"))
    ds = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/refs/heads/main/strongreject_dataset/strongreject_dataset.csv",
        cache_dir=str(DATA_DIR)+"/StrongREJECT",
    )['train']
    ds = ds.map(lambda x: {"source": "StrongREJECT-"+x.pop("source"), "question": x.pop("forbidden_prompt")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/StrongREJECT/StrongREJECT.csv", index=False)
    
    return ds

def load_JBBBehaviours() -> Dataset:
    if file_exists(str(DATA_DIR)+"/JBBBehaviours/JBBBehaviours.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/JBBBehaviours/JBBBehaviours.csv"))
    ds = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        name="behaviors",
        cache_dir=str(DATA_DIR)+"/JBBBehaviours",
    )["harmful"]
    ds = ds.map(lambda x: {"source": "JBBBehaviours-"+x.pop("Source"), "id": x.pop("Index"), "category": x.pop("Category"), "question": x.pop("Goal")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/JBBBehaviours/JBBBehaviours.csv", index=False)
    return ds

def load_SGBench() -> Dataset:
    if file_exists(str(DATA_DIR)+"/SGBench/SGBench.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/SGBench/SGBench.csv"))
    ds = load_dataset(
        "json",
        data_files="https://raw.githubusercontent.com/MurrayTom/SG-Bench/refs/heads/master/datasets/SG-Bench/original_query/malicious_instruction.json",
        cache_dir=str(DATA_DIR)+"/SGBench",
    )["train"]
    ds = ds.map(lambda x: {"source": "SGBench", "question": x.pop("query"), "category": re.search(r'\[\s*(.*?)\]', x.pop("safety_type")).group(1)[4:]})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/SGBench/SGBench.csv", index=False)
    return ds

def load_XSTest() -> Dataset:
    if file_exists(str(DATA_DIR)+"/XSTest/XSTest.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/XSTest/XSTest.csv"))
    ds = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv",
        cache_dir=str(DATA_DIR)+"/XSTest",
    )["train"]
    ds = ds.filter(lambda x: x["label"] == "safe")
    ds = ds.map(lambda x: {"source": "XSTest", "question": x.pop("prompt"), "id": x.pop("id")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/XSTest/XSTest.csv", index=False)
    return ds

def load_ForbiddenQ() -> Dataset:
    if file_exists(str(DATA_DIR)+"/ForbiddenQ/ForbiddenQ.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/ForbiddenQ/ForbiddenQ.csv"))
    ds = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/verazuo/jailbreak_llms/refs/heads/main/data/forbidden_question/forbidden_question_set.csv",
        cache_dir=str(DATA_DIR)+"/ForbiddenQ",
    )["train"]
    ds = ds.map(lambda x: {"source": "ForbiddenQ", "question": x.pop("question"), "id": x.pop("q_id"), "category": x.pop("content_policy_name")})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/ForbiddenQ/ForbiddenQ.csv", index=False)
    return ds

def load_WildChat() -> Dataset:
    if file_exists(str(DATA_DIR)+"/WildChat/WildChat.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/WildChat/WildChat.csv"))
    ds = load_dataset(
        "yuntian-deng/WildChat-1M-Full-with-parameters-internal",
        cache_dir=str(DATA_DIR)+"/WildChat",
    )["train"]
    ds = ds.filter(lambda x: x["toxic"] == True)
    ds = ds.filter(lambda x: x["turn"] == 1)
    ds = ds.filter(lambda x: x["language"] == "English")
    ds = ds.filter(lambda x: x["openai_moderation"][-1]["flagged"] == True)
    # ds = ds.map(lambda x: {"category": (k:=max(d:=x.pop("openai_moderation")[-1]["category_scores"], key=d.get)), "source": "WildChat", "category_score": d[k], "question": x.pop("conversation")[0]["content"], "id": x.pop("conversation_hash")})
    ds = ds.map(lambda x: {"category": (k:=max(d:=x.pop("openai_moderation")[-1]["category_scores"], key=d.get)), "source": "WildChat", "category_score": d[k], "question": x.pop("conversation")[0]["content"]})
    ds_all = []
    for harmful_type in ["harassment", "harassment/threatening", "hate", "hate/threatening", "self-harm", "self-harm/instructions", "self-harm/intent", "sexual", "sexual/minors", "violence", "violence/graphic"]:
        ds_filter = ds.filter(lambda x: x["category"] == harmful_type)
        sorted_ds_filter = ds_filter.sort("category_score", reverse=True)
        top_100_ds_filter = sorted_ds_filter.select(range(min(100, len(sorted_ds_filter))))
        ds_all.append(top_100_ds_filter)
        # print(harmful_type, f": {len(top_100_ds_filter)}") # debug
    ds = concatenate_datasets(ds_all)
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/WildChat/WildChat.csv", index=False)
    return ds

def load_wildjailbreak() -> Dataset:
    if file_exists(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"))
    ds = load_dataset(
        "allenai/wildjailbreak",
        name="eval",
        cache_dir=str(DATA_DIR)+"/wildjailbreak",
    )["train"]
    ds = ds.map(lambda x: {"source": "wildjailbreak", "question": x.pop("adversarial")})
    ds = ds.filter(lambda x: x["data_type"] == "adversarial_harmful")
    ds = ds.shuffle(seed=42).select(range(250))
    
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv", index=False)
    return ds

TEST_DATASETS = {
    # Name: [load function, selection function, #samples]

    # 313 samples
    "StrongREJECT": [load_StrongREJECT, None, None],
    # 100 samples
    "JBBBehaviours": [load_JBBBehaviours, None, None],
    # 1442 samples
    "SGBench": [load_SGBench, None, None],
    # 250 samples
    "wildjailbreak": [load_wildjailbreak, None, None],
    # # 370 samples (filter the toxic=False, turn!=1, language!=English, ["openai_moderation"][-1]["flagged"]!=True) (select top 100 samples for each harmful type, if less than 100, select all)
    # "WildChat": [load_WildChat, None, None],
    
    # 250 samples
    "XSTest": [load_XSTest, None, None],
}   

def load_test_datasets() -> Dataset:
    if file_exists(str(DATA_DIR)+"/test_datasets.csv"):
        return Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/test_datasets.csv"))
    ds_all = []
    for dataset in TEST_DATASETS:
        print(f"\tLoading {dataset} datasets...")
        ds = TEST_DATASETS[dataset][0]()
        ds_all.append(ds)
        print(ds)
    ds = concatenate_datasets(ds_all)
    df = ds.to_pandas()
    df.to_csv(str(DATA_DIR)+"/test_datasets.csv", index=False)
    return ds

def main():
    print("Loading test datasets...")
    ds = load_test_datasets()
    print(ds)
    print("Done.")
    # ds = load_wildjailbreak()
    # print(ds)
    
if __name__ == "__main__":
    main()