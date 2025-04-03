import argparse
import json, os
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import datetime


def get_r1_response(prompt_input, keys, patience=10, **gen_kwargs):

    while patience > 0:
        key = random.choice(keys)
        ds_client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = key
            )
        patience -= 1
        completion = None
        # print("debug", flush=True)
        try:
            completion = ds_client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[{"role": "user","content": prompt_input}],
                stream=False,
                **gen_kwargs,
                timeout=300
                )
            if completion.choices[0].finish_reason == "stop":
                return True, completion
            print(f'patience: {patience}, error: finish_reason {completion.choices[0].finish_reason}', f'key: {key}')
            time.sleep(5)
        except Exception as e:
            print(e, flush=True)
            print(f'key: {key}', flush=True)
            if completion:
                print(completion)
            time.sleep(5)    
    return False, ''

def unit_chat(keys, prompt_input, gen_kwargs):
    return get_r1_response(
        prompt_input,
        keys,
        **gen_kwargs
    )


def process_item(line, keys, gen_kwargs):
    if 'response' in line:
        return None  # Skip processing if already contains 'response'
    
    prompt = line["question"]
    success, completion = unit_chat(keys, prompt, gen_kwargs)
    record = {}
    
    if success:
        line["response"] = completion.choices[0].message.content
        record["finish_reason"] = completion.choices[0].finish_reason
        record["usage"] = {"completion_tokens": completion.usage.completion_tokens, "prompt_tokens": completion.usage.prompt_tokens, "total_tokens": completion.usage.total_tokens}
        record["completion_id"] = completion.id
        return line, record  # Return processed line

    return None  # Skip writing if processing failed

def main(keys):

    gen_kwargs = dict(
        temperature=0.6,
        top_p=0.7,
        max_tokens=8000,
    )

    save_pth = f"./datasets/star1_benign_ds_tmp.json"
    if os.path.exists(save_pth):
        ids = [json.loads(line)['id'] for line in open(save_pth)]
    else:
        ids = []
    print(f'got {len(ids)} sucessful items')

    all_data = json.load(open('./datasets/star1_benign.json'))
    filtered_data = [line for line in all_data if line['id'] not in ids][:1]
    # filtered_data = [line for line in all_data if line['id'] not in ids]
    print(f'got {len(filtered_data)} items to be generated, originally {len(all_data)} ')
    
    record4r1call_path = "./datasets/star1_benign_record4r1call.json"
    if os.path.exists(record4r1call_path):
        record4r1call = json.load(open(record4r1call_path))
    else:
        record4r1call = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, line, keys, gen_kwargs): line for line in filtered_data}
        
        pbar = tqdm(as_completed(futures), total=len(futures))
        
        for future in pbar:
            result, record4r1call_update = future.result()
            if result:
                with open(save_pth, 'a') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                if record4r1call_update:
                    record4r1call.append(record4r1call_update)
                    with open(record4r1call_path, 'w') as f:
                        json.dump(record4r1call, f)
            # Update tqdm description with the latest timestamp
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pbar.set_description(f"Last Update: {timestamp}")

    pbar.close()
    

if __name__=="__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs='+', type=str, help="List of API keys")
    parser.add_argument("--max_workers", type=int, help="Number of worker processes")
    args = parser.parse_args()
    
    main(keys=args.keys)
    
    save_pth = f"./datasets/star1_benign_ds_tmp.json"
    if os.path.exists(save_pth):
        items = [json.loads(line) for line in open(save_pth)]
    else:
        items = []
    with open("star1_benigh_with_cot.json", "w") as f:
        json.dump(items, f)