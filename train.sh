SAVE_DIR='./models/multitask/18_1p'

SAVE_DIR_CUR=${SAVE_DIR}
mkdir $SAVE_DIR_CUR
CUDA_VISIBLE_DEVICES='0,1' python train.py --loss_weights '1, 1' \
--attention_block False --cls_block False --regression_block True\
--decay_boundaries '15000, 20000' --lr_decay_factors '1, 0.1, 0.01' --max_number_of_steps 25000 \
--model_dir ${SAVE_DIR_CUR} \
# --checkpoint_path ${SAVE_DIR}/step1/model.ckpt-25000 --checkpoint_exclude_scopes 'ssd300/REG/'