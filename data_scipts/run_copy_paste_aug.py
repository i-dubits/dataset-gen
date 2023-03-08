#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from put_foreground_on_background import paste_not_overlap
import glob
from pathlib import Path
import random
from PIL import Image

# number of images to generate in each class
numb_of_aug_img = 200

# initial images folder
input_dir = 'masks'
# result images folder
output_dir = 'aug_copy_paste_5_classes'

# folder for background images
bg_dir = 'background_data'

input_paths = Path(input_dir)
input_subdirectories = [x for x in input_paths.iterdir() if x.is_dir()]

bg_files_paths = list(Path(bg_dir).iterdir())

for curr_dirr in input_subdirectories:
    
    # configure output path
    curr_label = str(curr_dirr).split('/')[-1]
    curr_output_dir = output_dir + '/' + curr_label
    Path(curr_output_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(numb_of_aug_img):
        # choose bg image
        bg_index = random.randint(0, len(bg_files_paths) - 1)
        
        # choose images to paste in bg image
        curr_imgs_paths = list(Path(curr_dirr).iterdir())
        nb_img_to_paste = random.randint(1, len(curr_imgs_paths))
        paths_to_paste = random.sample(curr_imgs_paths, nb_img_to_paste)
        imgs_to_paste = [Image.open(path) for path in paths_to_paste]
        
        output_path = curr_output_dir + '/' + curr_label + str(i) + '.png'
        paste_not_overlap(imgs_to_paste, bg_files_paths[bg_index], output_path)