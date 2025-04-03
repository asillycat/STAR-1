export data=$1 # strongreject / jbbbehaviours / wildchat / wildjailbreak / xstest
export model=$2 # DeepSeek-R1-Distill-Qwen-1.5B

python gen.py --data=${data} --model=${model} > gen_out/${data}/${model}.out 2>&1 &

wait