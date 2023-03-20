dataset=curved_ref
python test.py --name $dataset \
    --dataset_mode rffr --dataset_root ./load/rffr/${dataset} --img_wh 504 378 \
    --patch_size 4 --batch_size 128 \
    --model nerfren --mlp_network two_layer_mlp \
    --N_coarse 32 --N_importance 32 
    