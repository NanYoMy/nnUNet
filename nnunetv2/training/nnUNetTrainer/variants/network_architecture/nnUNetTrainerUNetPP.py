from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class UNetPlusPlus(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNetPlusPlus, self).__init__()
        self.encoder = nn.ModuleList([self._create_encoder_block(num_channels, 64),
                                      self._create_encoder_block(64, 128),
                                      self._create_encoder_block(128, 256),
                                      self._create_encoder_block(256, 512)])
        self.decoder = nn.ModuleList([self._create_decoder_block(512, 256),
                                      self._create_decoder_block(256, 128),
                                      self._create_decoder_block(128, 64),
                                      self._create_decoder_block(64, num_classes)])
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _create_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())

    def _create_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())

    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        for decoder_block in self.decoder:
            x = self.up(x)
            x = decoder_block(x)
            x = x + encoder_outputs.pop()
        return x

class nnUNetTrainerUNetPP(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        
        
        return UNetPlusPlus(num_channels=num_input_channels, num_classes=num_output_channels)
