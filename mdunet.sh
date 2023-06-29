export CUDA_VISIBLE_DEVICES=6;
nohup     python train.py --dataset ISIC2018   --arch MDUNet --name archs_mdunet_isic_6_11_19_06  --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 200 --input_w 512 --input_h 512 --b 4  --num_workers 16 --deep_supervision True > logs/train_mdunet_isic_6_11_19_06.log 2>&1 &
