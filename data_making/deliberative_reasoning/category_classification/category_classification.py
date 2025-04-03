import argparse
import json, os
from tqdm import tqdm
import time
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import datetime
from prompt import CLASSIFICATION_PROMPT_TEMPLATE, CATEGORY_DEFINITION

client_example = AzureOpenAI(
    azure_endpoint="",
    api_key = "",
    api_version=""
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
                if type(eval(response.choices[0].message.content)) != list:
                    continue
                return True, response
            
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


def process_item(line, gen_kwargs):
    if 'category' in line:
        return None, None  # Skip processing if already contains 'response'
    
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(instruction=line["question"], category_definitions=CATEGORY_DEFINITION)
    
    success, response = unit_chat(prompt, gen_kwargs)
    
    record = {}
    if success:
        line["category"] = eval(response.choices[0].message.content)
        # print(response)
        record["finish_reason"] = response.choices[0].finish_reason
        record["usage"] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens},
        record["completion_id"] = response.id
        return line, record  # Return processed line

    return None, None  # Skip writing if processing failed

def main():

    gen_kwargs = dict(
        max_completion_tokens=1024,
    )

    save_pth = f"./datasets/data_collection_ds_tmp.json"
    if os.path.exists(save_pth):
        ids = [json.loads(line)['id'] for line in open(save_pth)]
    else:
        ids = []
    print(f'got {len(ids)} sucessful items')
    

    all_data = json.load(open('./datasets/data_collection.json'))
    filtered_data = [line for line in all_data if line['id'] not in ids][:5]
    # filtered_data = [line for line in all_data if line['id'] not in ids]
    print(f'got {len(filtered_data)} items to be generated, originally {len(all_data)} ')
    
    record4r1call_path = "./datasets/data_collection_record4r1call.json"
    if os.path.exists(record4r1call_path):
        record4r1call = json.load(open(record4r1call_path))
    else:
        record4r1call = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, line, gen_kwargs): line for line in filtered_data}
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, help="Number of worker processes")
    args = parser.parse_args()
    main()
    
    save_pth = f"./datasets/data_collection_ds_tmp.json"
    if os.path.exists(save_pth):
        items = [json.loads(line) for line in open(save_pth)]
    else:
        items = []
    with open("data_with_category.json", "w") as f:
        json.dump(items, f)