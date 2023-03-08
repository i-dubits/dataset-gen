#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import random

def put_im_back(im_path, bg_path, output_path):
    '''Put image on background
    
    Args:
        im_path (String): path to foreground image
        bg_path (String): path ot background image
        output_path (String): path to store output image'''
    img = Image.open(im_path, 'r')
    bg = Image.open(bg_path, 'r')
    
    img_w, img_h = img.size
    bg_w, bg_h = bg.size
    
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    
    output_im = bg.copy()
    # https://stackoverflow.com/questions/5324647/how-to-merge-a-transparent-png-image-with-another-image-using-pil
    output_im.paste(img, (0,0), img)
    output_im.save(output_path)

# https://www.geeksforgeeks.org/find-two-rectangles-overlap/
def is_overlap(l1, r1, l2, r2):
    '''Check if two rectangles are overlapped
    
    Args:
        l1(Tuple):  top left x, y coordinates of the first rectangle
        r1(Tuple):  bottom right x, y coordinate of the first rectangle
        l2(Tuple):  top left x, y coordinates of the second rectangle
        r2(Tuple):  bottom right x, y coordinates of the second rectangle
    Returns: Bool: True - rectangles overlapped, False - rectangles are not overlapped'''    
    
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    return True

# https://stackoverflow.com/questions/54488217/mapping-an-image-with-random-coordinates-using-pil-without-them-stay-one-on-to
def paste_not_overlap(paste_image_list, bg_path, output_path, numb_attempts = 1000):
    '''Paste images from the list to background image
    
    If foreground images are overlapped another new random coordinates are chosen
    Args:
        paste_image_list (List[PIL.Image]): list of images to paste
        bg_path (String): path to background image
        output_path (String): path to save result image
        numb_attempts (Int): total number of attempts '''
    background = Image.open(bg_path)
    
    alread_paste_point_list = []
    
    for img in paste_image_list:
        # if all not overlap, find the none-overlap start point
        for _ in range(numb_attempts):
            # left-top point
            # x, y = random.randint(0, background.size[0]), random.randint(0, background.size[1])
    
            # if image need in the bg area, use this
            x, y = random.randint(0, max(0, background.size[0]-img.size[0])), random.randint(0, max(0, background.size[1]-img.size[1]))
    
            # right-bottom point
            l2, r2 = (x, y), (x+img.size[0], y+img.size[1])
    
            if all(not is_overlap(l1, r1, l2, r2) for l1, r1 in alread_paste_point_list):
                # save alreay pasted points for checking overlap
                alread_paste_point_list.append((l2, r2))
                background.paste(img, (x, y), img)
                break
    
    background.save(output_path)


