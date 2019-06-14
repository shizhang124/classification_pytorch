#!/usr/bin/python 
#coding=utf-8

import json, time

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
            if one_list or len(t)==1:
                rs.append(t[0])
            else:
                rs.append(t)
    return rs

def write_txt(dst_path, txt_list, df='\t'):
    with open(dst_path, 'r') as txt:
        for t in txt_list:
            ts = [str(tt) for tt in t]
            info = df.join(ts)
            txt_list.append(info+'\n')

