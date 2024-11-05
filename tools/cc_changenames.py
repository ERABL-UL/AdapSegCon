import json
import numpy as np
import os

with open('percentiles_split_KITTI360.json', 'r') as p:
    splits = json.load(p)
    points_datapath = []
    percentage = ['0.5', '0.2', '0.1', '0.02', '0.01']
    for percent in percentage:
        for seq in splits[percent]:
            for filename in splits[percent][seq]['data']:
                print(filename)
