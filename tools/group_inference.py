# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:13:15 2019

@author: lwj
"""

# encoding:utf/8
import sys
from mmdet.apis import inference_detector, init_detector
import json
import os
import numpy as np
import argparse
import pandas
from tqdm import tqdm
import pandas as pd
import time
import cv2
import torch
import torch.nn as nn
import pdb

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# generate result

def fuse_conv_bn(conv, bn):
    """ During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m


def result_from_dir():
    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')
    color = [(255, 0, 0), (0, 0, 0), (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    fused_model = fuse_module(model)
    pics = os.listdir(pic_path)[:1000]
    annotations = []
    num = 0
    time1 = time.time()
    missing_imgs = []
    eval_result = []
    for im in tqdm(pics):
        num += 1
        img_path = os.path.join(pic_path, im)
        result_ = inference_detector(fused_model, img_path)

        img = cv2.imread(img_path)

        for i, boxes in enumerate(result_, 1):
            if len(boxes):
                box_color = color[i]
                for box in boxes:
                    bbox = [round(float(i), 2) for i in box[0:4]]
                    conf = float(box[4])
                    annotations.append([im, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 'face', conf])
                    eval_result.append([img_path, i, conf, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))])
                    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, thickness = 2)
            else:
                missing_imgs.append([im])
        cv2.imwrite(os.path.join(csv_out_path, im), img)
    time2 = time.time()
    time_cost = time2 - time1
    print('Test FPS: {:.2f}'.format(len(pics) / time_cost))

    return eval_result, missing_imgs


def get_images_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    gt = []
    images_lst = []
    for item in json_dict['images']:
        flag = 0
        assert 'file_name' in item.keys()
        image_id = item['id']
        image_name = item['file_name']
        images_lst.append(image_name)
        for anno in json_dict['annotations']:
            if anno['image_id'] == image_id:
                label = anno['category_id']
                bbox = (
                anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3])
                gt.append([image_name, label, 1, bbox])
                flag = 1
        if flag == 0:
            print(image_name)

    return images_lst, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-m", "--model", help="Model path", type=str, )
    parser.add_argument("-c", "--config", help="Config path", type=str, )
    parser.add_argument("-im", "--im_dir", help="Image path", type=str, )
    parser.add_argument('-o', "--out", help="Save path", type=str, )
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    csv_out_path = args.out
    if not os.path.exists(csv_out_path):
        os.mkdir(csv_out_path)
    pic_path = args.im_dir

    eval_result, missing_imgs = result_from_dir()

