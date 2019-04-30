SAVE_DIR='./models/pupil'

SAVE_DIR_CUR=${SAVE_DIR}/step1-2
mkdir $SAVE_DIR_CUR
CUDA_VISIBLE_DEVICES='0,1' python train.py --loss_weights '1,1' \
--decay_boundaries '15000, 20000' --lr_decay_factors '1, 0.1, 0.01' --max_number_of_steps 5000 \
--model_dir ${SAVE_DIR_CUR}/ \
--checkpoint_path models/multitask/18_atten_9p/step3/model.ckpt-25000 --checkpoint_exclude_scopes 'ssd300/REG/second_loc'