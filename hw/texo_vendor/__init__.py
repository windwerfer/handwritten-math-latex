"""Vendored from https://github.com/alephpi/Texo (AGPL-3.0): HGNetv2 encoder.

SPDX-License-Identifier: AGPL-3.0-only

Registering it lets transformers' VisionEncoderDecoderModel load alephpi/FormulaNet.
"""
from transformers import AutoConfig, AutoModel
from .hgnet2 import HGNetv2, HGNetv2Config

AutoConfig.register("my_hgnetv2", HGNetv2Config)
AutoModel.register(HGNetv2Config, HGNetv2)
