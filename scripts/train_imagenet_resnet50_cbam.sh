sleep 3h
python train_imagenet.py --ngpu 2 --workers 8 --arch resnet --depth 18 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix ../RESNET18_IMAGENET_CBAM /Datasets/imagenet/
