import matplotlib.pyplot as plt
import numpy as np

from emq.models.mixed.resnet_imagenet import ResNet_ImageNet

# resnet18
m = ResNet_ImageNet(depth=18)

length = 18

bit_cfg = [8, 6, 8, 8, 7, 8, 8, 8, 8, 8, 4, 8, 6, 4, 4, 4, 4, 8]

FLOPs, first_last_flops = m.cfg2flops_layerwise(
    cfg=m.cfg, length=len(bit_cfg), quant_type='QAT')
print(FLOPs, first_last_flops)

print(f'bit_cfg = {bit_cfg} m_cfg = {m.cfg}')

print('Quantization model BOPs is',
      (((first_last_flops * 8 * 8 + sum(FLOPs[i] * bit_cfg[i] * 8
                                        for i in range(length)))) /
       (1024 * 1024 * 1024)), 'G')

# param
params, first_last_size = m.cfg2params_perlayer(
    cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')
params = [i / (1024 * 1024) for i in params]
first_last_size = first_last_size / (1024 * 1024)

print('Quantization model is',
      np.sum(np.array(bit_cfg) * np.array(params) / 8) + first_last_size, 'Mb')
# print('Original model is',
#       np.sum(np.array(params)) * 4 + first_last_size * 4, 'Mb')

# m = ResNet_ImageNet(depth=18)
# with open('./data/PTQ-GT.pkl', 'r') as f:
#     lines = f.readlines()
#     # record for plot
#     x_param = []
#     y_acc = []

#     for line in lines:
#         bit_cfg, acc = line.split(':')
#         bit_cfg = eval(bit_cfg)
#         acc = float(acc)
#         params, first_last_size = m.cfg2params_perlayer(
#             cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')
#         params = [i / (1024 * 1024) for i in params]
#         first_last_size = first_last_size / (1024 * 1024)
#         model_size = np.sum(
#             np.array(bit_cfg) * np.array(params) / 8) + first_last_size

#         x_param.append(model_size)
#         y_acc.append(acc)

#     plt.scatter(x_param, y_acc)
#     plt.xlabel('Model Size(Mb)')
#     plt.ylabel('Accuracy')
#     plt.show()
