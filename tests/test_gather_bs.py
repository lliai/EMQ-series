import os
import re
from glob import glob

# filter the files with BS in the filename from the root_path
# demo file: rnd_search_emq_zc_tree_resnet18_BS_64_2023_03_14_3333.log
root_path = './output/rnd_search_emq_zc'
files = glob(os.path.join(root_path, '*BS*.log'))

# read each file and get the spearman correlation from the last line
# last line:  * For EMQ: sp@top100%: 0.7270326115271462, sp@top50%: 0.4828780657548671, sp@top20%: 0.4666666666666666
# filter the spearman correlation: 0.7270326115271462

batchSize_info = {}

for file in files:
    bs = str(file.split('_')[-5])
    if bs not in batchSize_info:
        batchSize_info[bs] = []

    print('bs:', bs)
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith(' * For EMQ: sp@top100%:'):
                # use regular expression to extract floating point numbers
                pattern = re.compile(r'\d+\.\d+')
                # search for matches in the input string and append to the result list
                for i, match in enumerate(pattern.findall(line)):
                    print(i, '===', match)
                    if i == 0:
                        # first one
                        # if float(match) < 0.65:
                        #     continue
                        batchSize_info[bs].append(float(match))

# sort the dict by key
batchSize_info = dict(
    sorted(batchSize_info.items(), key=lambda item: int(item[0])))

print(batchSize_info)
