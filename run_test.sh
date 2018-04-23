source set_test_env.sh
python train.py --dataset_dir=$DATASET_DIR \
--train_dir=$TRAIN_DIR \
--checkpoint_path=$CHECKPOINT_PATH \
--labels_file=$LABELS_FILE \
--log_test=$TRAIN_DIR/test
