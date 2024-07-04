# for repeat
ab=false
for n_cls in 100    
do
    python3 main_lab_se.py --lr 0.1 --n_classes $n_cls --device 0 \
    --save_folder checkpoint/ab_loc_senet_cifar$n_cls --repeat 5 --epoch 200 \
    --batch_size 128 \
    # --ablation ${ab} 
done