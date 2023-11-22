DATASET=rcb
TEXT_COLUMN_NAME="premise,hypothesis"
CUR_TIME=`date +"%d-%m-%y_%T"`

python3 scripts/run_classification.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/${DATASET}/train.csv \
    --validation_file data/${DATASET}/test.csv \
    --metric_name accuracy \
    --text_column_name ${TEXT_COLUMN_NAME} \
    --text_column_delimiter «,» \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --output_dir tmp/${DATASET}_${CUR_TIME}/
