import face_pulse

rand_seeds = [
    42,
    100,
    300,
    500,
    700,
    900,
    1024,
    2048,
    4096,
    8192,
    # 10000,
    # 30000,
    # 50000,
    # 70000,
    # 90000,
    # 100000,
    # 300000,
    # 500000,
]
face_pulse.image_predict(rand_seeds, output_dir="output")
