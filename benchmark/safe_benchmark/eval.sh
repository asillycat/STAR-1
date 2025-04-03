export data=$1 # strongreject / jbbbehaviours / wildchat / wildjailbreak / xstest
export model=$2 # DeepSeek-R1-Distill-Qwen-1.5B
export eval_model=$3 # Llama-Guard / OpenAI4o_refusal

python eval.py --model=${eval_model} --part="response" --file_path="result/${model}/${data}_normal.json" > eval_out/${data}/${model}.out 2>&1 &

wait