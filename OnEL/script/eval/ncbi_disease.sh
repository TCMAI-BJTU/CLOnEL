TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "CLOnEL-NCBI-Disease" \
    --dataset_name_or_path "ncbi-disease" \
    --eval_dir "processed_test" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 45 \
    --topk 20 \
    --use_embed_parallel False
