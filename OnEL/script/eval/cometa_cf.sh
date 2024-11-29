TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "CLOnEL-COMETA-CF" \
    --dataset_name_or_path "cometa-cf" \
    --eval_dir "test.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 25 \
    --topk 20 \
    --use_embed_parallel
