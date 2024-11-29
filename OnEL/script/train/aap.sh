CUDA_VISIBLE_DEVICES=1 \
TOKENIZERS_PARALLELISM=false \
python train_aap.py \
    --model_name_or_path "SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "aap" \
    --train_dir "train.txt" \
    --eval_dir "test.txt" \
    --train_dictionary_path "test_dictionary.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 3 \
    --batch_size 16 \
    --epochs 2 \
    --learning_rate 5e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --use_tree_similarity
