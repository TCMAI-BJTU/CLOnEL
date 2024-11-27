TOKENIZERS_PARALLELISM=false \
python train.py \
    --model_name_or_path "/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/OnEL/checkpoints/ncbi-disease/model_upload" \
    --dataset_name_or_path "ncbi-disease" \
    --eval_dir "processed_test" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 45 \
    --topk 20 \
    --learning_rate 2e-5 \
    --use_embed_parallel False
