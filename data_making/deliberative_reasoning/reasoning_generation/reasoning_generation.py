import argparse
import json, os
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import datetime
from prompt import COT_PROMPT_TEMPLATE
from prompt import POLICY_HARASSMENT, POLICY_SEXUAL, POLICY_VIOLENCE, POLICY_SELF_HARM, POLICY_ILLICIT, POLICY_MISINFORMATION, POLICY_PRIVACY, POLICY_INTERLECTUAL

policy_map = {
    "Harassment / Hate / Discrimination": POLICY_HARASSMENT,
    "Sexual / Adult": POLICY_SEXUAL,
    "Violence / Physical Harm": POLICY_VIOLENCE,
    "Self-Harm": POLICY_SELF_HARM,
    "Illicit / Criminal Behavior": POLICY_ILLICIT,
    "Misinformation / Disinformation": POLICY_MISINFORMATION,
    "Privacy / Personal Data": POLICY_PRIVACY,
    "Intellectual Property": POLICY_INTERLECTUAL
}


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
    
    policy = policy_map[line["category"][0]]
    for category in line["category"][1:]:
        policy += "\n---\n" + policy_map[category]

    
    prompt = COT_PROMPT_TEMPLATE.format(instruction=line["question"], related_policies=policy)
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

    save_pth = f"./datasets/data_with_category_ds_tmp.json"
    if os.path.exists(save_pth):
        ids = [json.loads(line)['id'] for line in open(save_pth)]
    else:
        ids = []
    n_success = len(ids)
    print(f'got {len(ids)} sucessful items')
    
    # load errors
    error_pth = f"./datasets/data_with_category_ds_error_tmp.json"
    if os.path.exists(error_pth):
        ids += [json.loads(line)['id'] for line in open(error_pth)]

    print(f'got {len(ids)-n_success} error items')
    

    all_data = json.load(open('./datasets/data_with_category.json'))
    # filtered_data = [line for line in all_data if line['id'] not in ids][:5]
    filtered_data = [line for line in all_data if line['id'] not in ids]
    print(f'got {len(filtered_data)} items to be generated, originally {len(all_data)} ')
    
    record4r1call_path = "./datasets/data_with_category_record4r1call.json"
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
                if 'error' in result:
                    # Write to error output
                    with open(error_pth, 'a') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
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
    
    save_pth = f"./datasets/data_with_category_ds_tmp.json"
    if os.path.exists(save_pth):
        items = [json.loads(line) for line in open(save_pth)]
    else:
        items = []
    with open("data_with_cot.json", "w") as f:
        json.dump(items, f)