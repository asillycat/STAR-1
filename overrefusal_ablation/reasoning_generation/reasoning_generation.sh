export max_workers=1
export keys="key1 key2 key3"

python reasoning_generation.py \
    --max_workers $max_workers \
    --keys $keys