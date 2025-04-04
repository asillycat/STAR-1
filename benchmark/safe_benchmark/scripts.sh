mkdir gen_out
cd gen_out
mkdir strongreject
mkdir sgbench_bias
mkdir jbbbehaviours
mkdir wildchat
mkdir xstest
mkdir wildjailbreak
cd ..
mkdir eval_out
cd eval_out
mkdir strongreject
mkdir sgbench_bias
mkdir jbbbehaviours
mkdir wildchat
mkdir xstest
mkdir wildjailbreak

#  bash scripts.sh DeepSeek-R1-Distill-Qwen-1.5B

export model=$1 # DeepSeek-R1-Distill-Qwen-1.5B

export data=$2 # strongreject / jbbbehaviours / wildjailbreak / wildchat / xstest

# export data=strongreject
bash gen.sh $data ${model} &
# export data=jbbbehaviours
# bash gen.sh $data ${model} &
# export data=wildjailbreak
# bash gen.sh $data ${model} &
# export data=wildchat
# bash gen.sh $data ${model} &
# export data=xstest
# bash gen.sh $data ${model} &


# export data=strongreject
bash eval.sh $data ${model} Llama-Guard &
# export data=jbbbehaviours
# bash eval.sh $data ${model} Llama-Guard &
# export data=wildchat
# bash eval.sh $data ${model} Llama-Guard &
# export data=wildjailbreak
# bash eval.sh $data ${model} Llama-Guard &
# export data=xstest
# bash eval.sh $data ${model} OpenAI4o_refusal &

wait