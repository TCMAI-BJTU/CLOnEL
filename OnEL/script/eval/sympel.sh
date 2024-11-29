TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "CLOnEL-SYMPEL" \
    --dataset_name_or_path "sympel" \
    --eval_dir "test.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 20 \
    --topk 20 \
