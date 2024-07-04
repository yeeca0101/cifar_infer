# Resnext
# for repeat
for n_cls in 10 
do
    python3 main_lab_dense.py --lr 0.1 --n_classes $n_cls --device 0 \
    --save_folder checkpoint/densenet_cifar$n_cls --repeat 5 --epoch 200
done