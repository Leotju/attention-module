python train_imagenet.py \
			--ngpu 1 \
			--workers 8 \
			--arch resnet --depth 18 \
			--epochs 100 \
			--batch-size 256 --lr 0.1 \
			--att-type CBAM \
			--prefix ../RESNET18_IMAGENET_CBAM \
			/vpalab/diak_dataset/imagenet/


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py --ngpu 4 --workers 16 --arch resnet --depth 18 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix ../RESNET18_IMAGENET_CBAM /raid/tiancai/imagenet/imagenet/
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py --ngpu 4 --workers 16 --arch resnetbu --depth 18 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix ../RESNET18BU_IMAGENET_CBAM /raid/tiancai/imagenet/imagenet/

CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py --ngpu 2 --workers 8 --arch resnet_cascade --depth 18 --epochs 100 --batch-size 256 --lr 0.1 --att-type CAS_SE --prefix ../../chekpoint/RESNET18_IMAGENET_CAS_SE /Datasets/imagenet/

