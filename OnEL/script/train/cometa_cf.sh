CUDA_VISIBLE_DEVICES=2 \
TOKENIZERS_PARALLELISM=false \
python train.py \
    --model_name_or_path "SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "cometa-cf" \
    --train_dictionary_path "test_dictionary.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 3e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --use_embed_parallel \
    --use_tree_similarity
