source set_eval_env.sh
python -u eval.py --dataset_name=$DATASET_NAME \
--dataset_dir=$DATASET_DIR \
--dataset_split_name=validation \
--model_name=$MODLE_NAME \
--checkpoint_path=$TRAIN_DIR \
--eval_dir=$TRAIN_DIR/validation
