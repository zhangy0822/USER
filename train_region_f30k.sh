DATASET_NAME='f30k'
DATA_PATH='your dataset path/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --num_epochs=20 --lr_update=10 --learning_rate=.0005 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 \
  --batch_size 128 \
  --attention_lamda 1 \
  --use_moco 1 \
  --moco_M 2048 \
  --loss_lamda 20 \
  --mu 90 \
  --gama 0.5 \
  --moco_r 0.999 
