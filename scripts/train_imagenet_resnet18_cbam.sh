python train_imagenet.py \
			--ngpu 1 \
			--workers 8 \
			--arch resnet --depth 18 \
			--epochs 100 \
			--batch-size 256 --lr 0.1 \
			--att-type CBAM \
			--prefix ../RESNET18_IMAGENET_CBAM \
			/vpalab/diak_dataset/imagenet/
