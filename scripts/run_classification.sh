DATASET="ru_wanli"
TEXT_COLUMN_NAME="sentence"
CUR_TIME=`date +"%d-%m-%y_%T"`
STRATEGY="epoch"

python3 scripts/run_classification.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/ru_wanli/train_processed.csv \
    --validation_file data/ru_wanli/test_processed.csv \
    --test_files "rcb:data/rcb/train_processed.csv,terra:data/terra/train_processed.csv" \
    --metric_name accuracy \
    --text_column_name ${TEXT_COLUMN_NAME} \
    --text_column_delimiter «,» \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_train_samples 100 \
    --max_eval_samples 100 \
    --logging_strategy ${STRATEGY} \
    --evaluation_strategy ${STRATEGY} \
    --save_strategy ${STRATEGY} \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --output_dir tmp/${DATASET}_${CUR_TIME}/
