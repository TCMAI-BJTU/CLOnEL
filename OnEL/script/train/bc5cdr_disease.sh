CUDA_VISIBLE_DEVICES=2 \
TOKENIZERS_PARALLELISM=false \
python train.py \
    --model_name_or_path "SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "bc5cdr-disease" \
    --train_dir "processed_traindev" \
    --train_dictionary_path "train_dictionary.txt" \
    --eval_dir "processed_test" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 25 \
    --topk 20 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 1 \
    --use_embed_parallel \
    --use_tree_similarity
