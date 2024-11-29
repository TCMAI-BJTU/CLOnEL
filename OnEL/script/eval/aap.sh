CUDA_VISIBLE_DEVICES=3 \
TOKENIZERS_PARALLELISM=false \
python eval_aap.py \
    --model_name_or_path "CLOnEL-AAP" \
    --dataset_name_or_path "aap" \
    --eval_dir "test.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20