CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type minicpm-v-v2 \
    --model_id_or_path /home/Sw/OpenBMB/openbmb/MiniCPM-V-2 \
    --custom_train_dataset_path /home/Sw/LLM_learning/Mini_project_building/MiniCPM-Model-Train/data/MiniCPM_V2_Dataset.json \
    --batch_size 16 \
