export CUDA_VISIBLE_DEVICES=4

python3 main.py \
    --seed 2022 \
    --data <your_path>/imagenet \
    --data_v2 <your_path>/ImageNetV2 \
    --data_sketch <your_path>/imagenet-sketch \
    --data_corruption <your_path>/imagenet-c \
    --data_rendition <your_path>/imagenet-r \
    --data_adv <your_path>/imagenet-a \
    --output ./outputs \
    --model 'vitbase_timm' \
    --algorithm 'spa' \
    --tag '_test'