import argparse
from ast import arg
import pytz
from datetime import datetime
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from config import *
import re
import pandas as pd

import os

get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')

DATA_DIR = "datasets"

def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)

# dataset load and parsing
def parse_strongreject():

    data= load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
        cache_dir=str(DATA_DIR)+"/StrongREJECT",
    )["train"]

    harmful_inst = data['forbidden_prompt']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_jbbbehaviours():
    data = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        name="behaviors",
        cache_dir=str(DATA_DIR)+"/JBBBehaviours",
    )["harmful"]

    harmful_inst = data['Goal']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_wildchat():
    if file_exists(str(DATA_DIR)+"/WildChat/WildChat.csv"):
        data = Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/WildChat/WildChat.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "yuntian-deng/WildChat-1M-Full-with-parameters-internal",
        cache_dir=str(DATA_DIR)+"/WildChat",
    )["train"]
    data = data.filter(lambda x: x["toxic"] == True)
    data = data.filter(lambda x: x["turn"] == 1)
    data = data.filter(lambda x: x["language"] == "English")
    data = data.filter(lambda x: x["openai_moderation"][-1]["flagged"] == True)
    data = data.map(lambda x: {"category": (k:=max(d:=x.pop("openai_moderation")[-1]["category_scores"], key=d.get)), "category_score": d[k], "question": x.pop("conversation")[0]["content"]})
    ds_all = []
    for harmful_type in ["harassment", "harassment/threatening", "hate", "hate/threatening", "self-harm", "self-harm/instructions", "self-harm/intent", "sexual", "sexual/minors", "violence", "violence/graphic"]:
        ds_filter = data.filter(lambda x: x["category"] == harmful_type)
        sorted_ds_filter = ds_filter.sort("category_score", reverse=True)
        top_100_ds_filter = sorted_ds_filter.select(range(min(100, len(sorted_ds_filter))))
        ds_all.append(top_100_ds_filter)
        # print(harmful_type, f": {len(top_100_ds_filter)}") # debug
    data = concatenate_datasets(ds_all)
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/WildChat/WildChat.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_xstest() -> Dataset:
    if file_exists(str(DATA_DIR)+"/XSTest/XSTest.csv"):
        data =  Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/XSTest/XSTest.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "csv",
        data_files="https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv",
        cache_dir=str(DATA_DIR)+"/XSTest",
    )["train"]
    data = data.filter(lambda x: x["label"] == "safe")
    data = data.map(lambda x: {"question": x.pop("prompt")})
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/XSTest/XSTest.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_wildjailbreak() -> Dataset:
    if file_exists(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"):
        data =  Dataset.from_pandas(pd.read_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv"))
        harmful_inst = data['question']
        harmful_target = [None for _ in harmful_inst]
        return harmful_inst, harmful_target
    data = load_dataset(
        "allenai/wildjailbreak",
        name="eval",
        cache_dir=str(DATA_DIR)+"/wildjailbreak",
    )["train"]
    data = data.map(lambda x: {"source": "wildjailbreak", "question": x.pop("adversarial")})
    data = data.filter(lambda x: x["data_type"] == "adversarial_harmful")
    data = data.shuffle(seed=42).select(range(250))
    df = data.to_pandas()
    df.to_csv(str(DATA_DIR)+"/wildjailbreak/wildjailbreak.csv", index=False)
    harmful_inst = data['question']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def get_args():
    '''for resp_gen and pipeline'''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=RSM_LIST[0], choices=RSM_LIST)
    argparser.add_argument('--data', type=str, default='strongreject', choices=EVAL_DATA)
    argparser.add_argument('--system', type=bool, default=DEFAULT_GEN_CONFIG['system'])
    argparser.add_argument('--temperature', type=float, default=DEFAULT_GEN_CONFIG['temperature'])
    argparser.add_argument('--topp', type=float, default=DEFAULT_GEN_CONFIG['topp'])
    argparser.add_argument('--topk', type=int, default=DEFAULT_GEN_CONFIG['topk'])
    argparser.add_argument('--max_tokens', type=int, default=DEFAULT_GEN_CONFIG['max_tokens'])
    argparser.add_argument('--repeat_n', type=int, default=DEFAULT_GEN_CONFIG['repeat_n'], help='number of sample per prompt input')
    argparser.add_argument('--n', type=int, default=-1, help='number of samples to use, -1 to use all')
    argparser.add_argument('--start_idx', type=int, default=0, help='start index of the samples to use')
    args = argparser.parse_args()

    return args