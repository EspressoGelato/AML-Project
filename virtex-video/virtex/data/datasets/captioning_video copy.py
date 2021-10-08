import os
import random
from typing import Callable, Dict, List

import albumentations as alb
import numpy as np
import torch
from torch.utils.data import Dataset

from virtex.data.readers import LmdbReader
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T

from virtex.data.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from virtex.data.temporal_transforms import Compose as TemporalCompose
from virtex.data.get_mean_std import get_mean_std 

from virtex.data.spatial_transforms import (Compose, Normalize, Resize, CenterCrop, RandomCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

import argparse
from PIL import Image

SampleDuration = 16
n_val_samples = 3
opt = {}
opt['train_crop_min_scale'] = 0.25
opt['train_crop_min_ratio'] = 0.75
opt['mean_dataset'] = 'kinetics'
opt['value_scale'] = 1
opt['mean'], opt['std'] = get_mean_std(opt['value_scale'], opt['mean_dataset'])
opt['no_mean_norm'] = None
opt['no_std_norm'] = None
opt['sample_size'] = 112
opt = argparse.Namespace(**opt)
 

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


class CaptioningDatasetVideo(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a serialized LMDB file (COCO Captions in this codebase). This is used for
    pretraining tasks which use captions - bicaptioning, forward captioning and
    token classification.

    This dataset also supports training on a randomly selected subset of the
    full dataset.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain the serialized
        LMDB files (for COCO ``train2017`` and ``val2017`` splits).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    tokenizer: virtex.data.tokenizers.SentencePieceBPETokenizer
        A tokenizer which has the mapping between word tokens and their
        integer IDs.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    max_caption_length: int, optional (default = 30)
        Maximum number of tokens to keep in output caption tokens. Extra tokens
        will be trimmed from the right end of the token list.
    use_single_caption: bool, optional (default = False)
        COCO Captions provides five captions per image. If this is True, only
        one fixed caption per image is use fo training (used for an ablation).
    percentage: float, optional (default = 100.0)
        Randomly sample this much percentage of full dataset for training.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
    ):
        lmdb_path = os.path.join(data_root, f"serialized_{split}.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("<unk>")


        temporal_transform = []

        if split == 'train':
            temporal_transform.append(TemporalRandomCrop(SampleDuration))
            temporal_transform = TemporalCompose(temporal_transform)

        else:
            temporal_transform.append(TemporalRandomCrop(SampleDuration))
                #TemporalEvenCrop(SampleDuration, n_val_samples))

            temporal_transform = TemporalCompose(temporal_transform)
        
        self.temporal_transform = temporal_transform

        spatial_transform = [] 
        normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
        if split == 'train':
            spatial_transform.append(
                        RandomResizedCrop(
                        opt.sample_size, (opt.train_crop_min_scale, 1.0),
                        (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))

            spatial_transform.append(RandomHorizontalFlip())

            spatial_transform.append(ToTensor())
            spatial_transform.append(ScaleValue(opt.value_scale))
            spatial_transform.append(normalize)
            spatial_transform = Compose(spatial_transform) 
        else:
            spatial_transform = [
                Resize(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor()
                ]
            spatial_transform.extend([ScaleValue(opt.value_scale), normalize])  
            spatial_transform = Compose(spatial_transform)

        self.spatial_transform = spatial_transform
  


    def __len__(self):
        return len(self.reader)

    def __loading(self, image_list, frame_indices):
        clip = [image_list[idx] for idx in frame_indices]#self.loader(path, frame_indices)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(Image.fromarray(img)) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        image_id, image_list, captions = self.reader[idx]
        image_id = int(image_id[5:])

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        #image_caption = self.image_transform(image=image, caption=caption)
        #image, caption = image_caption["image"], image_caption["caption"]
        #image = np.transpose(image, (2, 0, 1))

        frame_indices = list(range(len(image_list)))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = self.__loading(image_list, frame_indices)
        #print(clip.shape)

        caption_tokens = self.caption_transform(caption=captions)["caption"]
        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": clip.float(),#torch.tensor(clip, dtype=torch.float),
            "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
            "noitpac_tokens": torch.tensor(caption_tokens, dtype=torch.long).flip(0),
            "caption_lengths": torch.tensor(len(caption_tokens), dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["caption_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["noitpac_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "caption_tokens": caption_tokens,
            "noitpac_tokens": noitpac_tokens,
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
