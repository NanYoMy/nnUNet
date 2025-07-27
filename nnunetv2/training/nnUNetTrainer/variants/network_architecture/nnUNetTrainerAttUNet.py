from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network import AttU_Net

class nnUNetTrainerAttUNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 200

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        
        
        return AttU_Net(img_ch=num_input_channels, output_ch=num_output_channels)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        # if self.is_ddp:
        #     mod = self.network.module
        # else:
        #     mod = self.network
        # if isinstance(mod, OptimizedModule):
        #     mod = mod._orig_mod

        # mod.decoder.deep_supervision = enabled
        pass