# for repeat
for n_cls in 100 10
do
    python3 main_lab.py --lr 0.1 --n_classes $n_cls --device 1 --save_folder checkpoint/dev_GLUs_v3_cifar$n_cls --repeat 5 --epoch 200
done