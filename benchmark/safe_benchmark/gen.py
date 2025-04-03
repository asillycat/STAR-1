import os
import json
import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import *
from utils import *

def file_path_gen(args, output_dir='result'):
    file_dir = f'{output_dir}/{args.model.split("/")[-1]}'
    if os.path.exists(file_dir) is False:
        os.makedirs(file_dir)

    file_path = f'{file_dir}/{args.data}'
    for param in DEFAULT_GEN_CONFIG:
        # save changed parameters
        if getattr(args, param) != DEFAULT_GEN_CONFIG[param]:
            file_path += f'_{param}-{getattr(args, param)}'

    file_path += f'.json'
    
    return file_path

def load_dataset(args):
    dataset_name = args.data
    n = args.n
    if dataset_name == 'strongreject':
        data = parse_strongreject()
    elif dataset_name == "jbbbehaviours":
        data = parse_jbbbehaviours()
    elif dataset_name == "wildchat":
        data = parse_wildchat()
    elif dataset_name == "xstest":
        data = parse_xstest()
    elif dataset_name == "wildjailbreak":
        data = parse_wildjailbreak()
    else:
        raise ValueError('Unknown dataset name')

    if n != -1:
        print(f'Using only {n} samples')
        data = (data[0][args.start_idx:n], data[1][args.start_idx:n])
    return data

def input_proc(inst, target, tokenizer, args, system_prompt, run_api=False):
    msg = []
    if system_prompt and args.system:
        msg.append({"role": "system", "content": system_prompt})
    
    msg.append({"role": "user", "content": inst})
    if not run_api:
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                                                    msg,
                                                    tokenize=False,
                                                    add_generation_prompt=True
                                                )
        else:
            text = f'''## Instruction: {inst}\n## Response: '''
        return text
    else:
        return msg
        

def main(args):
    harmful_inst, harmful_target = load_dataset(args)
    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG[args.model]['model_path'])
    context_length = args.max_tokens

    model =  LLM(model=MODEL_CONFIG[args.model]['model_path'], 
                trust_remote_code=True,
                tensor_parallel_size=MODEL_CONFIG[args.model]['n_gpu'],
                max_num_seqs=64,
                gpu_memory_utilization=0.95,
                dtype = torch.bfloat16
            )

    
    prompts = []
    for inst, target in zip(harmful_inst, harmful_target):
        text = input_proc(inst, target, tokenizer, args, MODEL_CONFIG[args.model]['system_prompt'])
        prompts.append(text)

    print('vLLM generation stage')
    
    sampling_params = SamplingParams(temperature=args.temperature,
                                        top_p=args.topp,
                                        top_k=args.topk,
                                        max_tokens= context_length,
                                        n = args.repeat_n,
                                        stop_token_ids=[tokenizer.eos_token_id],
                                        repetition_penalty=1.2)
    
    res_data = []

    resp = model.generate(prompts, sampling_params=sampling_params)
    for inst, prompt, resp_out in zip(harmful_inst, prompts,  resp):
        data = {
            'instruction': inst,
            'prompt': prompt,
            'response': [n_out.text.strip() for n_out in resp_out.outputs]
        }
        res_data.append(data)
    
    # save tp json
    meta_data = vars(args)

    saved_data = {'meta_info': meta_data, 'data': res_data}

    file_path = file_path_gen(args)
    with open(file_path, 'w') as f:
        json.dump(saved_data, f, indent=4)
    print('Results saved to', file_path)
    return file_path


if __name__ == '__main__':
    args = get_args()
    print(args)
    file_path = main(args)
