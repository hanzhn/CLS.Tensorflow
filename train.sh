CUDA_VISIBLE_DEVICES='0,1' python train.py --loss_weights '1, 2, 0' --decay_boundaries '15000, 20000' --lr_decay_factors='1, 0.1, 0.01' --max_number_of_steps 25000

# CUDA_VISIBLE_DEVICES='0,1' python train.py --loss_weights '0, 0, 10' --decay_boundaries '30000, 35000' --lr_decay_factors '1, 0.1, 0.01' --max_number_of_steps 40000
