import os
import re

root_path = './output/rank_naive_zc'

zc2time_sp_dict = {}
# zc2time_sp_dict = {
#    'qe': {
#    'time': [],
#    'sp100': [0.3],
#    'sp20': [0.4],
#    'sp50': [0.5],
# }

# traverse all ".log" files
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.log'):
            infos = file.split('_')
            if len(infos) < 4:
                continue
            zc_name = infos[3]
            full_path = os.path.join(root, file)
            # open .log file and read it line by line
            with open(full_path, 'r') as f:
                for line in f:
                    if '*' in line and 'Time' in line:
                        if zc_name in zc2time_sp_dict:
                            if 'time' in zc2time_sp_dict[zc_name]:
                                zc2time_sp_dict[zc_name]['time'].append(
                                    float(line.split(':')[-1].rstrip('s\n')))
                            else:
                                zc2time_sp_dict[zc_name]['time'] = [
                                    float(line.split(':')[-1].rstrip('s\n'))
                                ]
                        else:
                            zc2time_sp_dict[zc_name] = {}
                            zc2time_sp_dict[zc_name]['time'] = [
                                float(line.split(':')[-1].rstrip('s\n'))
                            ]

                    if '*' in line and 'sp' in line:
                        # example: '* For qe_score: sp@top100%: 0.35, sp@top50%: 0.32, sp@top20%: 0.34'
                        # sp100 = 0.35, sp50 = 0.32, sp20 = 0.34
                        result_list = []

                        # use regular expression to extract floating point numbers
                        pattern = re.compile(r'\d+\.\d+')

                        # search for matches in the input string and append to the result list
                        for match in pattern.findall(line):
                            result_list.append(float(match))

                        print(zc_name, result_list)

                        if zc_name in zc2time_sp_dict:
                            if 'sp100' in zc2time_sp_dict[zc_name]:
                                zc2time_sp_dict[zc_name]['sp100'].append(
                                    result_list[0])
                                zc2time_sp_dict[zc_name]['sp50'].append(
                                    result_list[1])
                                zc2time_sp_dict[zc_name]['sp20'].append(
                                    result_list[2])
                            else:
                                zc2time_sp_dict[zc_name] = {
                                    'sp100': [result_list[0]],
                                    'sp50': [result_list[1]],
                                    'sp20': [result_list[2]],
                                }
                        else:
                            zc2time_sp_dict[zc_name] = {
                                'sp100': [result_list[0]],
                                'sp50': [result_list[1]],
                                'sp20': [result_list[2]],
                            }

# process zc2time_sp_dict
print(zc2time_sp_dict)

# save dict as yaml
import yaml

with open('zc2time_sp_dict.yaml', 'w') as f:
    yaml.dump(zc2time_sp_dict, f)

# save dict as csv
import csv

for zc_name, time_sp in zc2time_sp_dict.items():
    with open(f'{zc_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'sp100', 'sp50', 'sp20'])
        for i in range(len(time_sp['time'])):
            writer.writerow([
                time_sp['time'][i], time_sp['sp100'][i], time_sp['sp50'][i],
                time_sp['sp20'][i]
            ])
