OUTPUT_DIR='/cache/zy/internvl_dpo_debug'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


torchrun internvl/train/internvl_chat_finetune_dpo.py \
  --model_name_or_path "/cache/zy/model/InternVL2-4B" \
  --conv_style "phi3-chat" \
  --output_dir ${OUTPUT_DIR} \
  --data_path /cache/data/dpo_train_data.jsonl \
  --image_folder /cache/data/image \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 1 \
  --bf16 False \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed path/to/ds_config_zero2.json \
  --report_to "tensorboard" \
  --fp16 True \
  2>&1 | tee -a "/cache/zy/internvl_dpo_debug/training_log.txt"
