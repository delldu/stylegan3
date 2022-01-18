import face_gan3

rand_seeds = [
    128,
    256,
    # 1024,
    # 2048,
]
# face_gan3.sample(rand_seeds, output_dir="output")
# face_gan3.project("images/dell.png", "output/dell.png")
# face_gan3.factorize()
# face_gan3.factorize_weight()

face_gan3.sefa(rand_seeds, output_dir="output")

