import csv
import os
import re

import yaml

# root_path = './output/rank_naive_zc'
root_path = './output/rnd_search_emq_zc'

zc2rank_dict = {}

# traverse all ".log" files
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.log'):
            infos = file.split('_')
            if len(infos) < 7:
                continue
            zc_name = infos[2]
            full_path = os.path.join(root, file)
            # open .log file and read it line by line
            with open(full_path, 'r') as f:
                for line in f:
                    if '*' in line and 'kd' in line:
                        # example: ' * For hawqv2: kd: 0.5845296201624779, ps: 0.7786908790069502, sp: 0.7880505404539005'
                        # kd=0.5845296201624779, ps=0.7786908790069502, sp=0.7880505404539005
                        result_list = []

                        # use regular expression to extract floating point numbers
                        pattern = re.compile(r'\d+\.\d+')

                        # search for matches in the input string and append to the result list
                        for match in pattern.findall(line):
                            result_list.append(float(match))

                        if len(result_list) < 3:
                            continue

                        print(zc_name, result_list)

                        if zc_name in zc2rank_dict:
                            if 'kd' in zc2rank_dict[zc_name]:
                                zc2rank_dict[zc_name]['kd'].append(
                                    result_list[0])
                                zc2rank_dict[zc_name]['ps'].append(
                                    result_list[1])
                                zc2rank_dict[zc_name]['sp'].append(
                                    result_list[2])
                            else:
                                zc2rank_dict[zc_name] = {
                                    'kd': [result_list[0]],
                                    'ps': [result_list[1]],
                                    'sp': [result_list[2]],
                                }
                        else:
                            zc2rank_dict[zc_name] = {
                                'kd': [result_list[0]],
                                'ps': [result_list[1]],
                                'sp': [result_list[2]],
                            }

# process zc2rank_dict
print(zc2rank_dict)

# save dict as yaml

with open('zc2rank_dict.yaml', 'w') as f:
    yaml.dump(zc2rank_dict, f)

# save dict as csv

for zc_name, time_sp in zc2rank_dict.items():
    with open(f'{zc_name}_rank.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['kd', 'ps', 'sp'])
        for i in range(len(time_sp['kd'])):
            writer.writerow(
                [time_sp['kd'][i], time_sp['ps'][i], time_sp['sp'][i]])
