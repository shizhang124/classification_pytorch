#!/usr/bin/python
# coding=utf-8

import json
import time
import random
import sys
import os


class Time:
    def __init__(self):
        self.start = time.time()
        self.end = time.time()
        self.use = self.end - self.start

    def tick(self, info=''):
        self.end = time.time()
        self.use = self.end - self.start
        self.start = time.time()
        print('--time--', info, ':', round(self.use, 1), 's')


def load_txt(src_path, df=None, one_list=False):
    rs = []
    with open(src_path, 'r') as txt:
        for line in txt:
            if df is None:
                t = line.strip().split()
            else:
                t = line.strip().split(df)
            if one_list or len(t) == 1:
                rs.append(t[0])
            else:
                rs.append(t)
    return rs


def write_txt(dst_path, txt_list, df='\t'):
    with open(dst_path, 'r') as txt:
        for t in txt_list:
            ts = [str(tt) for tt in t]
            info = df.join(ts)
            txt_list.append(info + '\n')


def load_json(src_path):
    dst_dict = {}
    with open(src_path, 'r') as f:
        dst_dict = json.load(f)
    return dst_dict


def write_json(dst_path, dst_dict):
    with open(dst_path, 'w') as f:
        json.dump(dst_dict, f)


def load_gt_bbox(src_path):
    pic_dict = {}
    src_txt = load_txt(src_path)
    len_bbox = 5
    for t in src_txt:
        num = (len(t) - 1) // len_bbox
        pic_name = t[0]
        bbox_list = []
        for i in range(num):
            x1 = int(t[1 + i * len_bbox])
            y1 = int(t[2 + i * len_bbox])
            x2 = int(t[3 + i * len_bbox])
            y2 = int(t[4 + i * len_bbox])
            label = t[5 + i * len_bbox]
            bbox_list.append([x1, y1, x2, y2, label])
        if pic_name not in pic_dict:
            pic_dict[pic_name] = []
        else:
            pic_name[pic_name].append(bbox_list)
    return pic_dict


def load_pre_bbox(src_path, prob_thre=0):
    pic_dict = {}
    src_txt = load_txt(src_path)
    len_bbox = 6
    for t in src_txt:
        num = (len(t) - 1) // len_bbox
        pic_name = t[0]
        bbox_list = []
        for i in range(num):
            x1 = int(t[1 + i * len_bbox])
            y1 = int(t[2 + i * len_bbox])
            x2 = int(t[3 + i * len_bbox])
            y2 = int(t[4 + i * len_bbox])
            label = t[5 + i * len_bbox]
            prob = float(t[6 + i * len_bbox])
            if prob >= prob_thre:
                bbox_list.append([x1, y1, x2, y2, label, prob])
        if pic_name not in pic_dict:
            pic_dict[pic_name] = []
        else:
            pic_name[pic_name].append(bbox_list)
    return pic_dict


def xyxy2ltwh(data):
    bbox = [int(t) for t in data]
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    return bbox


def ltwh2xyxy(data):
    bbox = [int(t) for t in data]
    bbox[2] = bbox[2] + bbox[0]
    bbox[3] = bbox[3] + bbox[1]
    return bbox

if __name__ == "__main__":
    print("test txt_tool")
    print("test txt_tool")