import torch.nn as nn

from emq.quant.fold_bn import search_fold_and_remove_bn
from emq.quant.quant_block import BaseQuantBlock, specials
from emq.quant.quant_layer import QuantModule, StraightThrough


class QuantModel(nn.Module):
    """Covert a PyTorch model to QuantModel

    Args:
        model (nn.Module): _description_
        weight_quant_params (dict, optional): _description_. Defaults to {}.
        act_quant_params (dict, optional): _description_. Defaults to {}.
    """

    def __init__(self,
                 model: nn.Module,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {}):

        super().__init__()
        search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params,
                                   act_quant_params)

    def quant_module_refactor(self,
                              module: nn.Module,
                              weight_quant_params: dict = {},
                              act_quant_params: dict = {}):
        """Recursively replace the normal conv2d and Linear layer to QuantModule

        Args:
            :param module: nn.Module with nn.Conv2d or nn.Linear in
                its children
            :param weight_quant_params: quantization parameters like n_bits
                for weight quantizer
            :param act_quant_params: quantization parameters like n_bits
                for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(
                    module, name,
                    specials[type(child_module)](child_module,
                                                 weight_quant_params,
                                                 act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module, name,
                    QuantModule(child_module, weight_quant_params,
                                act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params,
                                           act_quant_params)

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def set_bias_state(self,
                       use_bias_corr: bool = False,
                       vcorr_weight: bool = False,
                       bcorr_weight: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.set_bias_state(use_bias_corr, vcorr_weight, bcorr_weight)

    def forward(self, input):
        return self.model(input)

    def forward_before_global_avg_pool(self, input):
        return self.model.forward_before_global_avg_pool(input)

    def forward_with_features(self, input):
        return self.model.forward_with_features(input)

    def madnas_forward_pre_GAP(self):
        return self.model.madnas_forward_pre_GAP()

    def get_stage_info(self):
        return self.model.get_stage_info()

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]

        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def set_mixed_precision(self, bit_cfg):
        bit_cfgs = []
        if len(bit_cfg) == 8:
            for i in range(len(bit_cfg)):
                for j in range(2):
                    bit_cfgs.append(bit_cfg[i])
                if i in [2, 4, 6]:
                    bit_cfgs.append(bit_cfg[i])
        elif len(bit_cfg) == 20:
            for i in range(len(bit_cfg)):
                if i == 0 or i == 18 or i == 19:
                    bit_cfgs.append(bit_cfg[i])
                elif i == 1:
                    for j in range(2):
                        bit_cfgs.append(bit_cfg[i])
                else:
                    for j in range(3):
                        bit_cfgs.append(bit_cfg[i])
        elif len(bit_cfg) == 53:
            bit_cfgs = bit_cfg
        else:
            for i in range(len(bit_cfg)):
                bit_cfgs.append(bit_cfg[i])
                if i in [6, 10, 14]:
                    bit_cfgs.append(bit_cfg[i])
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]

        for i in range(len(bit_cfgs)):
            module_list[i].weight_quantizer.bitwidth_refactor(bit_cfgs[i])
