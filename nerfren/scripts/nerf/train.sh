dataset=$1
python train.py --name $1_nerf \
    --dataset_mode rffr --dataset_root ../datasets/$1 --img_wh 504 378 \
    --patch_size 1 --batch_size 2048 \
    --model nerf --mlp_network vanilla_mlp \
    --N_coarse 32 --N_importance 32 
    