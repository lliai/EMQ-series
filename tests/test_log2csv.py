import csv
import os

log_path = '/home/stack/project/EMQ/output/evo_search_emq_zc/evo_search_emq_zc_tree_resnet18_2023_03_02_981006_ablation_DPS.log'
csv_path = '/home/stack/project/EMQ/output/evo_search_emq_zc/evo_search_emq_zc_tree_resnet18_2023_03_02_981006_ablation_DPS.csv'


def convert_log2csv(log_path, csv_path):
    # read log file line by line and extract the index and sp and convert it into csv
    with open(log_path, 'r') as log_f, open(csv_path, 'w') as csv_f:
        csv_writer = csv.writer(csv_f)
        # csv_writer.writerow(['index', 'KD'])
        for line in log_f:
            if line.startswith('Iter'):
                index = int(line.split(' ')[1])
                sp = float(line.split(' ')[4])
                csv_writer.writerow([index, sp])


if __name__ == '__main__':
    root_dir = '/home/stack/project/EMQ/output/evo_search_emq_zc'
    # filter all .log files
    for file in os.listdir(root_dir):
        if file.endswith('.log') and 'tree' in file:
            log_path = os.path.join(root_dir, file)
            csv_path = os.path.join(root_dir, file.replace('.log', '.csv'))
            convert_log2csv(log_path, csv_path)
            # print(log_path, csv_path)
