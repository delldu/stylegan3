python gen_images.py --outdir=out --trunc=0.75 --seeds=$1-$2 \
    --noise-mode=random \
    --network=checkpoints/stylegan3-r-ffhq-1024x1024.pkl

# python gen_images.py --outdir=out --trunc=0.7 --seeds=$1-$2 \
#     --noise-mode=random \
#     --network=checkpoints/stylegan3-r-ffhqu-256x256.pkl
