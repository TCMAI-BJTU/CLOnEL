TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "CLOnEL-BC5CDR-Chemical" \
    --dataset_name_or_path "bc5cdr-chemical" \
    --eval_dir "processed_test" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20 \
    --use_embed_parallel