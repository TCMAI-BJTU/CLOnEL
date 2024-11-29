TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "CLOnEL-BC5CDR-Disease" \
    --dataset_name_or_path "bc5cdr-disease" \
    --eval_dir "processed_test" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 25 \
    --topk 20 \
    --use_embed_parallel False
