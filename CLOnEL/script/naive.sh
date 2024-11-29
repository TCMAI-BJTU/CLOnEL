CUDA_VISIBLE_DEVICES=2 \
TOKENIZERS_PARALLELISM=false \
python train.py \
    --model_name_or_path "/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/pretrain_model/SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "./data/sympel-cl" \
    --max_length 20 \
    --topk 20 \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --n_experiences 10 \
    --cl_strategy naive \
    --tree_ratio 0.5 \
    --use_tree_similarity

