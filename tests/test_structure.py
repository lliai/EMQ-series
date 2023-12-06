import emq.models.hubconf as hubconf  # noqa: F401, F403
from emq.api import EMQAPI
from emq.dataset.imagenet import build_imagenet_data
from emq.operators.zc_inputs import *  # noqa: F401, F403
from emq.quant import QuantModel
from emq.structures import GraphStructure, LinearStructure, TreeStructure

train_loader, test_loader = build_imagenet_data(
    batch_size=8, workers=0, data_path='E://all_imagenet_data')

cnn = eval('hubconf.resnet18(pretrained=False)')
cnn.eval()
# print('Quantized accuracy before brecq: {}'.format(validate_model(test_loader, cnn)))
# build quantization parameters
wq_params = {'n_bits': 4, 'channel_wise': False, 'scale_method': 'mse'}
aq_params = {
    'n_bits': 4,
    'channel_wise': False,
    'scale_method': 'mse',
    'leaf_param': False
}
qnn = QuantModel(
    model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
qnn.eval()
bit_cfg = [3, 4, 3, 3, 3, 2, 4, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2]
emqapi = EMQAPI('./data/PTQ-GT.pkl', verbose=False)

img, label = next(iter(train_loader))

struct = LinearStructure(length=4)
struct = TreeStructure(n_nodes=4)
struct = GraphStructure(n_nodes=3)

print('struct1:', struct)

zc = struct(img, label, qnn, bit_cfg)
print('zc:', zc)

other = LinearStructure(length=4)
other = TreeStructure(n_nodes=4)
other = GraphStructure(n_nodes=3)

print('struct2:', other)

new_struct = struct.cross_over_by_genotype(other)
print('cross over:', new_struct)
new_struct = struct.mutate_by_genotype()
print('mutate:', new_struct)
