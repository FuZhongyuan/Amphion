# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset factory for VC models.
Supports multiple datasets: emilia, ljspeech, libritts, etc.
"""

from models.vc.base.vc_emilia_dataset import VCEmiliaDataset, VCCollator
from models.vc.base.vc_ljspeech_dataset import VCLJSpeechDataset, VCLJSpeechCollator
from models.vc.base.vc_libritts_dataset import VCLibriTTSDataset, VCLibriTTSCollator


def get_vc_dataset_class(cfg):
    """
    Get VC dataset class based on configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        dataset_class: Dataset class
        collator_class: Collator class
    """
    # Check dataset type from config
    dataset_type = getattr(cfg.preprocess, "dataset_type", None)
    
    # If not specified, try to infer from dataset ratios
    if dataset_type is None:
        if hasattr(cfg, "dataset"):
            # Check if dataset is a dict-like object
            dataset_dict = cfg.dataset
            if isinstance(dataset_dict, dict):
                if "libritts" in dataset_dict and dataset_dict.get("libritts", 0) > 0:
                    dataset_type = "libritts"
                elif "ljspeech" in dataset_dict and dataset_dict.get("ljspeech", 0) > 0:
                    dataset_type = "ljspeech"
                elif "emilia" in dataset_dict and dataset_dict.get("emilia", 0) > 0:
                    dataset_type = "emilia"
            else:
                # Try to access as attributes
                if hasattr(dataset_dict, "libritts") and getattr(dataset_dict, "libritts", 0) > 0:
                    dataset_type = "libritts"
                elif hasattr(dataset_dict, "ljspeech") and getattr(dataset_dict, "ljspeech", 0) > 0:
                    dataset_type = "ljspeech"
                elif hasattr(dataset_dict, "emilia") and getattr(dataset_dict, "emilia", 0) > 0:
                    dataset_type = "emilia"
    
    # Default to libritts if still not determined
    if dataset_type is None:
        # Check use_emilia_dataset flag for backward compatibility
        if hasattr(cfg.train, "use_emilia_dataset") and cfg.train.use_emilia_dataset:
            dataset_type = "emilia"
        else:
            dataset_type = "libritts"  # Default to libritts
    
    # Return appropriate dataset class
    if dataset_type.lower() == "libritts":
        return VCLibriTTSDataset, VCLibriTTSCollator
    elif dataset_type.lower() == "ljspeech":
        return VCLJSpeechDataset, VCLJSpeechCollator
    elif dataset_type.lower() == "emilia":
        return VCEmiliaDataset, VCCollator
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Supported types: 'emilia', 'ljspeech', 'libritts'"
        )

