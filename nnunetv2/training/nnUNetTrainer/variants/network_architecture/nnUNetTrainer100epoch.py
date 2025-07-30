from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network import AttU_Net

class nnUNetTrainer100epoch(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

