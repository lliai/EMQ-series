import torch
import torch.nn as nn

from . import measure


class LinearRegionCount(object):
    """Computes and stores the average and current value"""

    def __init__(self, n_samples, gpu=None):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None
        self.gpu = gpu

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron)
            if self.gpu is not None:
                self.activations = self.activations.cuda(self.gpu)
        self.activations[self.ptr:self.ptr + n_batch] = torch.sign(
            activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(),
                           (1 - self.activations).T.half())
        res += res.T
        res = 1 - torch.sign(res)
        res = res.sum(1)
        res = 1. / res.float()
        self.n_LR = res.sum().item()
        del self.activations, res
        self.activations = None
        if self.gpu is not None:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:

    def __init__(self,
                 models=[],
                 input_size=(64, 3, 32, 32),
                 gpu=None,
                 sample_batch=1,
                 dataset=None,
                 data_path=None,
                 seed=0,
                 device=None):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        # self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.gpu = gpu
        self.device = torch.device('cuda:{}'.format(
            self.gpu)) if self.gpu is not None else torch.device('cpu')
        self.device = device if device is not None else self.device
        # print('Using device:{}'.format(self.device))

        self.reinit(models, input_size, sample_batch, seed)

    def reinit(self,
               models=None,
               input_size=None,
               sample_batch=None,
               seed=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [
                LinearRegionCount(
                    self.input_size[0] * self.sample_batch, gpu=self.gpu)
                for _ in range(len(models))
            ]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                # self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch

        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            if self.gpu is not None:
                torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [
            LinearRegionCount(self.input_size[0] * self.sample_batch)
            for _ in range(len(self.models))
        ]
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            inputs = torch.randn(self.input_size, device=self.device)

            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            # model.forward(input_data.cuda())
            model.forward(input_data)
            if len(self.interFeature) == 0:
                return
            feature_data = torch.cat(
                [f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)


@measure('linear_region', bn=True)
def compute_linear_region_number(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    lrc_model = Linear_Region_Collector(
        models=[net], input_size=inputs.shape, gpu=0, sample_batch=1)
    num_linear_regions = float(lrc_model.forward_batch_sample()[0])
    del lrc_model
    # torch.cuda.empty_cache()
    return num_linear_regions
