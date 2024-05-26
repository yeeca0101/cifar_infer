# for shuffleNetv2

n_cls=10

for net_size in 2. 
do
    python3 main_lab_shfv2.py --lr 0.1 --n_classes $n_cls --device 0 --save_folder checkpoint/shfv2_${net_size}_cifar$n_cls --repeat 5 --epoch 200
done



# for n_cls in 10 
# do
#     python3 main_lab.py --lr 0.1 --n_classes $n_cls --device 0 --save_folder checkpoint/dev_cifar$n_cls --repeat 5 --epoch 200
# done