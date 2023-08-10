import os
from tqdm import tqdm
import argparse
import csv
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default="../results", type=str)
    parser.add_argument('--method', default="ddp", type=str)
    parser.add_argument('--log_2', type=str)
    parser.add_argument('--log_4', type=str)
    parser.add_argument('--log_8', type=str)
    parser.add_argument('--log_16', type=str)
    parser.add_argument('--log_32', type=str)
    parser.add_argument('--log_64', type=str)

    args = parser.parse_args()

    return args

args = parse_args()

files = glob.glob(os.path.join(args.result, '*.log'))

max_ids = {}

for file in files:
    filename = os.path.basename(file)
    if args.method == 'ddp' and ('hvd' in filename or 'ds' in filename):
        continue 
    elif args.method != 'ddp' and (args.method not in filename):
        continue

    parts = filename.split('.')[0].split('_')

    nodes = int(parts[1])
    if args.method == 'ddp':
        id = int(parts[2])
    else:
        id = int(parts[3])

    if nodes not in max_ids or id > max_ids[nodes][0]:
        max_ids[nodes] = (id, file)

records = []
for num_nodes, (id, file) in max_ids.items():
    

    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            if "throughput" not in line:
                continue

            record = {}
            record['method'] = args.method
            record['num_nodes'] = num_nodes
            json_str = line.split(':::MLLOG ')[1]
            json_data = json.loads(json_str)
            record['Throughput'] = json_data['value']['throughput']

            records.append(record)

records = sorted(records, key=lambda x: x['num_nodes'])

keys = records[0].keys()
with open(f'{args.method}.csv', 'w') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(records)