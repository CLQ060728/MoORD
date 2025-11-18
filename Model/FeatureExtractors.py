from typing import List, Literal
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPModel, AutoModel
from PIL.Image import Image
import logging


logger = logging.getLogger(__name__)


class ClipFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, as_linear_classifier: bool = False,
                 chkpt_dir: str="./pre_trained/OPENAI_CLIP/"):
        """
        params:
            device: torch.device, device to run the model on
            as_linear_classifier: bool, whether to use the model as a linear classifier
            chkpt_dir: str, directory of the pre-trained CLIP model
        """
        super(ClipFeatureExtractor, self).__init__()
        self.dino_model = CLIPModel.from_pretrained(chkpt_dir).vision_model
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        # self.dino_model.config.output_hidden_states = True  # Enable output hidden states
        self.dino_model.to(device)
        self.device = device
        self.as_linear_classifier = as_linear_classifier
        if self.as_linear_classifier:
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1)
            )
            self.head.requires_grad_(True)

    def forward(self, imgs: torch.Tensor | List[Image] | Image):
        if isinstance(imgs, List):
            imgs = torch.stack([self.transform_img_clip()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = self.transform_img_clip()(imgs).unsqueeze(0).float().to(self.device)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image"
                             + " or a torch.Tensor.")
        
        if not self.as_linear_classifier:
            return self.dino_model(imgs)
        else:
            features = self.dino_model(imgs)
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
            return self.head(features)

    def transform_img_clip(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                            interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])


class Dinov2FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, as_linear_classifier: bool = False,
                 chkpt_dir: str="./pre_trained/DINO_V2/"):
        """
        params:
            device: torch.device, device to run the model on
            as_linear_classifier: bool, whether to use the model as a linear classifier
            chkpt_dir: str, directory of the pre-trained DINO V2 model
        """
        super(Dinov2FeatureExtractor, self).__init__()
        # self.processor = AutoImageProcessor.from_pretrained(chkpt_dir, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(chkpt_dir)
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        self.dino_model.to(device)
        self.device = device
        self.as_linear_classifier = as_linear_classifier
        if self.as_linear_classifier:
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1)
            )
            self.head.requires_grad_(True)

    def forward(self, imgs: torch.Tensor | List[Image] | Image):
        if isinstance(imgs, List):
            imgs = torch.stack([self.transform_img_dino()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = self.transform_img_dino()(imgs).unsqueeze(0).float().to(self.device)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image"
                             + " or a torch.Tensor.")
        
        # if not isinstance(imgs, torch.Tensor):
        #     inputs = self.processor(images=imgs, return_tensors="pt")
        # else:
        inputs = imgs

        if not self.as_linear_classifier:
            return self.dino_model(inputs)
        else:
            features = self.dino_model(inputs)
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
            return self.head(features)
        

    def transform_img_dino(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                            interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])


class Dinov3FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, pre_ds: Literal["LVD", "SAT"] = "LVD",
                 as_linear_classifier: bool = False,
                 chkpt_dir: str="./pre_trained/DINO_V3/"):
        """
        params:
            device: torch.device, device to run the model on
            pre_ds: Literal["LVD", "SAT"], pre-training dataset, defaults to "LVD-1689M", the other is "SAT-493M"
            as_linear_classifier: bool, whether to use the model as a linear classifier
            chkpt_dir: str, directory of the pre-trained DINO V3 model
        """
        super(Dinov3FeatureExtractor, self).__init__()
        if pre_ds == "LVD":
            chkpt_dir = chkpt_dir + "LVD/"
        elif pre_ds == "SAT":
            chkpt_dir = chkpt_dir + "SAT/"
        else:
            raise ValueError("Unknown pre-training dataset.")
        
        # self.processor = AutoImageProcessor.from_pretrained(chkpt_dir, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(
            chkpt_dir,
            device_map=device # "auto"
        )
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        self.dino_model.to(device)
        self.device = device
        self.pre_ds = pre_ds
        self.as_linear_classifier = as_linear_classifier
        if self.as_linear_classifier:
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1)
            )
            self.head.requires_grad_(True)

    def forward(self, imgs: torch.Tensor | List[Image] | Image):
        if self.pre_ds == "LVD":
            transform_img = self.transform_img_lvd
        elif self.pre_ds == "SAT":
            transform_img = self.transform_img_sat

        if isinstance(imgs, List):
            imgs = torch.stack([transform_img()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = transform_img()(imgs).unsqueeze(0).float().to(self.device)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image"
                             + " or a torch.Tensor.")

        # if not isinstance(imgs, torch.Tensor):
        #     inputs = self.processor(images=imgs, return_tensors="pt")
        # else:
        inputs = imgs

        if not self.as_linear_classifier:
            return self.dino_model(inputs)
        else:
            features = self.dino_model(inputs)
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
            return self.head(features)

    def transform_img_lvd(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

    def transform_img_sat(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])
