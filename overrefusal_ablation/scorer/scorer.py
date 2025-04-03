import argparse
import json, os
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import datetime
from openai import AzureOpenAI
from prompt import OVERREFUSAL_SCORER_PROMPT_TEMPLATE, OVERREFUSAL_SCORER_FORMAT_EXAMPLE
import re

client_example = AzureOpenAI(
    azure_endpoint= "",
    api_key = "",
    api_version= ""
)

clients = [
    client_example
]

def get_4o_response(prompt_input, patience=3, **gen_kwargs):

    while patience > 0:
        client = random.choice(clients)
        patience -= 1
        response = None
        try:
            response = client.chat.completions.create(
                model="gpt-4o-1120-nofilter-global",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_input}
                        ]
                    }
                ],
                **gen_kwargs
            )
            
            if response.choices[0].finish_reason == "stop":
                return True, response
            else:
                print(patience, response)
        except Exception as e:
            print(e)
            if response:
                print(response)
            time.sleep(5)    
    return False, ''

def unit_chat(prompt_input, gen_kwargs):
    return get_4o_response(
        prompt_input,
        **gen_kwargs
    )

def extract_cot_and_answer(text):
    match = re.search(r'<think>\n?(.*?)\n?</think>\n?\n?(.*)', text, re.DOTALL)
    if match:
        cot = match.group(1).strip()
        final_answer = match.group(2).strip()
        return cot, final_answer
    print("error")
    return None, None

def process_item(line, gen_kwargs):
    if 'score' in line:
        return None, None  # Skip processing if already contains 'response'

    cot, answer = extract_cot_and_answer(line['response'])
    
    prompt = OVERREFUSAL_SCORER_PROMPT_TEMPLATE.format(
        instruction=line["question"],
        cot=cot,
        response=answer,
        format_example=OVERREFUSAL_SCORER_FORMAT_EXAMPLE,
    )
    
    success, response = unit_chat(prompt, gen_kwargs)
    # print(response)
    record = {}
    if success:
        text = response.choices[0].message.content
        match = re.search(r"```json([\s\S]*?)```", text)
        if match:
            line["score"] = eval(match.group(1).strip())
            # print(response)
            record["finish_reason"] = response.choices[0].finish_reason
            record["usage"] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens},
            record["completion_id"] = response.id
            return line, record  # Return processed line
        else:
            return {'error': 'no json found'}, None

    return None, None  # Skip writing if processing failed

def main():

    gen_kwargs = dict(
        # temperature=0.6,
        # top_p=0.7,
        max_completion_tokens=1024,
    )

    save_pth = f"./datasets/star1_benigh_with_cot_ds_tmp.json"
    if os.path.exists(save_pth):
        ids = [json.loads(line)['id'] for line in open(save_pth)]
    else:
        ids = []
    n_success = len(ids)
    print(f'got {len(ids)} sucessful items')
    
    # load errors
    error_pth = f"./datasets/star1_benigh_with_cot_ds_error_tmp.json"
    if os.path.exists(error_pth):
        ids += [json.loads(line)['id'] for line in open(error_pth)]

    print(f'got {len(ids)-n_success} error items')
    

    all_data = json.load(open('./datasets/star1_benigh_with_cot.json'))
    filtered_data = [line for line in all_data if line['id'] not in ids][:5]
    # filtered_data = [line for line in all_data if line['id'] not in ids]
    print(f'got {len(filtered_data)} items to be generated, originally {len(all_data)} ')
    
    record4r1call_path = "./datasets/star1_benigh_with_cot_record4r1call.json"
    if os.path.exists(record4r1call_path):
        record4r1call = json.load(open(record4r1call_path))
    else:
        record4r1call = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, line, gen_kwargs): line for line in filtered_data}
        
        pbar = tqdm(as_completed(futures), total=len(futures))
        
        for future in pbar:
            result, record4r1call_update = future.result()
            # print(result)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, help="Number of worker processes")
    args = parser.parse_args()
    main()
    
    save_pth = f"./datasets/star1_benigh_with_cot_ds_tmp.json"
    if os.path.exists(save_pth):
        items = [json.loads(line) for line in open(save_pth)]
    else:
        items = []
    with open("star1_benigh_with_score.json", "w") as f:
        json.dump(items, f)
        
        
    # get full-score samples
    all_5 = []
    for item in items:
        flag = True
        for _, score in item["score"].items():
            if score != 5:
                flag = False
                break
        if flag:
            item['source'] = '4o_rewrite'
            all_5.append(item)
            
    with open("star1_benigh.json", 'w', encoding="utf-8") as f:
        json.dump(all_5, f, indent=4)
