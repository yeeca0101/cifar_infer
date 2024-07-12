# for shuffleNetv2

n_cls=200


python3 main_tiny.py --lr 0.01 --n_classes $n_cls --device 1  --batch_size 128\
    --save_folder checkpoint/resnet18_tiny --repeat 5 --epoch 200



# for n_cls in 10 
# do
#     python3 main_lab.py --lr 0.1 --n_classes $n_cls --device 0 --save_folder checkpoint/dev_cifar$n_cls --repeat 5 --epoch 200
# done