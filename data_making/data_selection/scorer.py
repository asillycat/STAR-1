import argparse
import json, os
from tqdm import tqdm
import time
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import datetime
from prompt import SCORER_PROMPT_TEMPLATE, SCORER_FORMAT_EXAMPLE
from prompt import POLICY_HARASSMENT, POLICY_SEXUAL, POLICY_VIOLENCE, POLICY_SELF_HARM, POLICY_ILLICIT, POLICY_MISINFORMATION, POLICY_PRIVACY, POLICY_INTERLECTUAL
import re

client_example = AzureOpenAI(
    azure_endpoint="",
    api_key = "",
    api_version=""
)

clients = [
    client_example
]

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
    
    policy = policy_map[line["category"][0]]
    for category in line["category"][1:]:
        policy += "\n---\n" + policy_map[category]

    cot, answer = extract_cot_and_answer(line['response'])
    prompt = SCORER_PROMPT_TEMPLATE.format(
        instruction=line["question"],
        cot=cot,
        response=answer,
        related_policies=policy,
        format_example=SCORER_FORMAT_EXAMPLE,
    )
    
    success, response = unit_chat(prompt, gen_kwargs)

    record = {}
    if success:
        text = response.choices[0].message.content
        match = re.search(r"```json([\s\S]*?)```", text)
        if match:
            line["score"] = eval(match.group(1).strip())

            record["finish_reason"] = response.choices[0].finish_reason
            record["usage"] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens},
            record["completion_id"] = response.id
            return line, record  # Return processed line
        else:
            return {'error': 'no json found'}, None

    return None, None  # Skip writing if processing failed

def main():

    gen_kwargs = dict(
        max_completion_tokens=1024,
    )

    save_pth = f"./datasets/data_with_cot_ds_tmp.json"
    if os.path.exists(save_pth):
        ids = [json.loads(line)['id'] for line in open(save_pth)]
    else:
        ids = []
    n_success = len(ids)
    print(f'got {len(ids)} sucessful items')
    
    # load errors
    error_pth = f"./datasets/data_with_cot_ds_error_tmp.json"
    if os.path.exists(error_pth):
        ids += [json.loads(line)['id'] for line in open(error_pth)]

    print(f'got {len(ids)-n_success} error items')
    

    all_data = json.load(open('./datasets/data_with_cot.json'))
    filtered_data = [line for line in all_data if line['id'] not in ids][:5]
    # filtered_data = [line for line in all_data if line['id'] not in ids]
    print(f'got {len(filtered_data)} items to be generated, originally {len(all_data)} ')
    
    record4r1call_path = "./datasets/data_with_cot_record4r1call.json"
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
    
    save_pth = f"./datasets/data_with_cot_ds_tmp.json"
    if os.path.exists(save_pth):
        items = [json.loads(line) for line in open(save_pth)]
    else:
        items = []
    with open("data_with_score.json", "w") as f:
        json.dump(items, f)
    
    # get full-score samples
    all_10 = []
    for item in items:
        flag = True
        if "Other" in item["category"]:
            continue
        for policy, score in item["score"].items():
            if score != 10:
                flag = False
                break
        if flag:
            all_10.append(item)
            
    with open("all_10.json", 'w', encoding="utf-8") as f:
        json.dump(all_10, f, indent=4)

    with open("all_10.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1k data seletion (remain diversity)
    while len(data) > 1000:
        category_num = {}
        categories = ["Harassment / Hate / Discrimination", "Sexual / Adult", "Violence / Physical Harm", "Self-Harm", "Illicit / Criminal Behavior", "Misinformation / Disinformation", "Privacy / Personal Data", "Intellectual Property"]
        sum_count = 0
        for category in categories:
            category_data = [item for item in data if category in item["category"]]
            category_num[category] = len(category_data)
            sum_count += len(category_data)

        category_prob = {}
        for category in categories:
            category_prob[category] = category_num[category] / sum_count

        source_prob = {}
        sources = ['GPTFuzz', 'SimpleSafetyTests', 'MaliciousInstruct', 'QHarm', 'TDCRedTeaming',
        'MaliciousInstructions', 'HarmfulQ', 'HExPHI', 'HarmBench', 'AttaQ',
        'HarmfulQA', 'SafetyInstructions', 'UltraSafety', 'ALERT', 'SaladBench',
        'PKUSafeRLHF', 'BeaverTails', 'wildjailbreak']
        for source in sources:
            source_data = [item for item in data if source == item["source"]]
            source_prob[source] = len(source_data) / len(data)

        item_prob = [source_prob[item['source']]*sum([category_prob[category] for category in item['category']]) for item in data]
        max_index = item_prob.index(max(item_prob))
        data.remove(data[max_index])
        
    with open("star1_high.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

