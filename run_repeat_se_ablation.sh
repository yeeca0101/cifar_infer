batch_size=16
# for batch_size in 16 8
# do
    for n_cls in 10
    do
        python3 main_lab_se.py --lr 0.01 --n_classes $n_cls --device 0 \
        --save_folder checkpoint/ab_senet_cifar${n_cls}_bs${batch_size} --repeat 5 --epoch 200 \
        --batch_size ${batch_size}
    done
# done
