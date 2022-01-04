python gen_images.py --outdir=out --trunc=0.7 --seeds=$1-$2 \
    --noise-mode=random \
    --network=checkpoints/stylegan3-r-ffhq-1024x1024.pkl

# python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=1x1 \
#     --w-frames=5 \
#     --network=checkpoints/stylegan3-r-ffhq-1024x1024.pkl
