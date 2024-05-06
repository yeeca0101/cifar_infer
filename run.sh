
for n_cls in 100 10
do
    python3 main.py --lr 0.1 --n_classes $n_cls --device 0 --save_folder checkpoint/cifar$n_cls
done


