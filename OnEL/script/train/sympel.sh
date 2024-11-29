CUDA_VISIBLE_DEVICES=2 \
TOKENIZERS_PARALLELISM=false \
python train.py \
    --model_name_or_path "SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "sympel" \
    --train_dir "train.txt" \
    --train_dictionary_path "test_dictionary.txt" \
    --eval_dir "test.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 20 \
    --topk 20 \
    --batch_size 8 \
    --epochs 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --tree_ratio 0.5 \
    --use_tree_similarity
