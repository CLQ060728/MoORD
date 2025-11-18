from torch.utils.data import Dataset
import torch
import lightning as LTN
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Literal, Dict, Tuple
import os, logging, json

logger = logging.getLogger(__name__)


class DFDDFMDataset(Dataset):
    def __init__(self,
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 img_root_path: str,
                 manifolds_paths: Tuple[str, str],
                 task_mode: Literal["ntest", "test"] = "ntest",
                 sem_idxes_path: str = None,
                 manifolds_indices_correspondence: Dict[str, str] = None):
        """
            params:
                model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model to be used. Options are "CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT".
                img_root_path: str The root path to the images.
                manifolds_paths: Tuple[str, str] The paths to the manifolds.
                task_mode: Literal["ntest", "test"] The mode of the task. Options are "ntest" for non-test mode and "test" for test mode.
                sem_idxes_path: str The path to the semantic indices mapping file. Required if task_mode is "test".
                manifolds_indices_correspondence: Dict[str, str] The correspondence between the indices of the manifolds.
        """
        assert os.path.exists(img_root_path), f"Image root path {img_root_path} does not exist."
        for path in manifolds_paths:
            manifold_path = os.path.join(img_root_path, path)
            assert os.path.exists(manifold_path), f"Manifold path {manifold_path} does not exist."
        assert isinstance(manifolds_paths, tuple) and all(isinstance(x, str) for x in manifolds_paths), f"manifolds_paths should be a tuple of strings, but got {type(manifolds_paths)}."
        # assert isinstance(manifolds_labels, tuple) and all(isinstance(x, int) for x in manifolds_labels), f"manifolds_labels should be a tuple of integers, but got {type(manifolds_labels)}."
        assert len(manifolds_paths) == 2, f"manifolds_paths should contain exactly two paths, but got {len(manifolds_paths)}."
        assert manifolds_paths[0] != manifolds_paths[1], f"manifolds_paths should be different."
        # assert len(manifolds_labels) == 2, f"manifolds_labels should contain exactly two labels, but got {len(manifolds_labels)}."
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"], f"model_type should be one of ['CLIP', 'DINO_V2', 'DINO_V3_LVD', 'DINO_V3_SAT'], but got {model_type}."
        assert task_mode in ["ntest", "test"], f"task_mode should be one of ['ntest', 'test'], but got {task_mode}."
        if task_mode == "test":
            assert sem_idxes_path is not None and sem_idxes_path != "", f"sem_idxes_path should not be None or '' when task_mode is 'test', but got {sem_idxes_path}."
        else:
            assert sem_idxes_path is None or sem_idxes_path == "", f"sem_idxes_path should be None or '' when task_mode is 'ntest', but got {sem_idxes_path}."
        assert isinstance(manifolds_indices_correspondence, dict), \
            f"manifolds_indices_correspondence should be a dict, but got {type(manifolds_indices_correspondence)}."

        mani_0_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[0])))
        mani_1_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[1])))
        
        logger.debug(f"{manifolds_paths[0]} images paths sample: {mani_0_imgs_paths[:10]}")
        logger.debug(f"{manifolds_paths[1]} images paths sample: {mani_1_imgs_paths[:10]}")
        logger.debug(f"{manifolds_paths[0]} images paths shape: {mani_0_imgs_paths.shape}")
        logger.debug(f"{manifolds_paths[1]} images paths shape: {mani_1_imgs_paths.shape}")

        assert len(mani_0_imgs_paths) == len(mani_1_imgs_paths) and len(mani_0_imgs_paths) % 2 == 0, \
            f"{manifolds_paths[0]} and {manifolds_paths[1]} images must have the same number of images "\
          + f"(even number), but got {len(mani_0_imgs_paths)} and {len(mani_1_imgs_paths)}."

        pair_indices_length = len(mani_0_imgs_paths) // 2
        mani_0_idx_imgs_dict = {}
        for img_path in mani_0_imgs_paths:
            cls = img_path.split("_")[0]
            if cls not in mani_0_idx_imgs_dict:
                mani_0_idx_imgs_dict[cls] = [f"{manifolds_paths[0]}/{img_path}"]
            else:
                mani_0_idx_imgs_dict[cls].append(f"{manifolds_paths[0]}/{img_path}")
        mani_1_idx_imgs_dict = {}
        for img_path in mani_1_imgs_paths:
            cls = img_path.split("_")[0]
            if cls not in mani_1_idx_imgs_dict:
                mani_1_idx_imgs_dict[cls] = [f"{manifolds_paths[1]}/{img_path}"]
            else:
                mani_1_idx_imgs_dict[cls].append(f"{manifolds_paths[1]}/{img_path}")

        logger.debug(f"mani_0_idx_imgs_dict number of keys: {len(mani_0_idx_imgs_dict)}")
        logger.debug(f"mani_1_idx_imgs_dict number of keys: {len(mani_1_idx_imgs_dict)}")

        mani_0_1_pair = []
        mani_1_0_pair = []
        labels_0_1 = []
        labels_1_0 = []
        mani2_0_1 = []
        mani2_1_0 = []
        for key, imgs in mani_0_idx_imgs_dict.items():
            logger.debug(f"class: {key}")

            mani_1_imgs = mani_1_idx_imgs_dict[manifolds_indices_correspondence[key]]

            logger.debug(f"corresponding class: {manifolds_indices_correspondence[key]}")

            mani_0_imgs = np.random.choice(imgs, size=len(imgs), replace=False)
            mani_1_imgs = np.random.choice(mani_1_imgs, size=len(mani_1_imgs), replace=False)

            logger.debug(f"shuffled {key}; mani_0_imgs[:6]: {mani_0_imgs[:6]}")
            logger.debug(f"shuffled {manifolds_indices_correspondence[key]}; mani_1_imgs[:6]: {mani_1_imgs[:6]}")
            logger.debug(f"mani_0_imgs shape: {mani_0_imgs.shape}; mani_1_imgs shape: {mani_1_imgs.shape}")

            mani_0_1_pair.extend(list(zip(mani_0_imgs[:len(mani_0_imgs)//2], mani_1_imgs[:len(mani_1_imgs)//2])))
            mani_1_0_pair.extend(list(zip(mani_1_imgs[len(mani_1_imgs)//2:], mani_0_imgs[len(mani_0_imgs)//2:])))
            
            labels_0_1.extend(np.array([[0], [1]], dtype=np.float32).reshape(1,
                            2, 1).repeat(len(mani_0_imgs)//2, axis=0).tolist())
            labels_1_0.extend(np.array([[1], [0]], dtype=np.float32).reshape(1,
                            2, 1).repeat(len(mani_1_imgs)//2, axis=0).tolist())
            
            mani2_0_1.extend(np.array([1]).repeat(len(mani_0_imgs)//2, axis=0).tolist())
            mani2_1_0.extend(np.array([0]).repeat(len(mani_1_imgs)//2, axis=0).tolist())

            logger.debug(f"mani_0_1_pair shape: {np.array(mani_0_1_pair).shape}")
            logger.debug(f"mani_1_0_pair shape: {np.array(mani_1_0_pair).shape}")
            logger.debug(f"labels_0_1 shape: {np.array(labels_0_1).shape}")
            logger.debug(f"labels_1_0 shape: {np.array(labels_1_0).shape}")
            logger.debug(f"mani2_0_1 shape: {np.array(mani2_0_1).shape}")
            logger.debug(f"mani2_1_0 shape: {np.array(mani2_1_0).shape}")

        self.data = np.concatenate([mani_0_1_pair, mani_1_0_pair], axis=0)

        logger.debug(f"Data sample (first 10 pairs): {self.data[:10]}")
        logger.debug(f"Data sample (second 10 pairs): {self.data[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of pairs before shuffling: {self.data.shape}")

        data_indices = np.arange(self.data.shape[0])

        logger.debug(f"data_indices before shuffling (first top 10): {data_indices[:10]}")
        logger.debug(f"data_indices before shuffling (second top 10): {data_indices[pair_indices_length:pair_indices_length+10]}")

        np.random.shuffle(data_indices)
        self.data = self.data[data_indices]

        logger.debug(f"data_indices after shuffling (first top 10): {data_indices[:10]}")
        logger.debug(f"data_indices after shuffling (second top 10): {data_indices[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Data sample (first 10 pairs): {self.data[:10]}")
        logger.debug(f"Data sample (second 10 pairs): {self.data[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of pairs after shuffling: {self.data.shape}")

        self.manifold2_indices = np.concatenate([mani2_0_1, mani2_1_0], axis=0).astype(np.int64)

        logger.debug(f"Manifold 2 indices sample (first 10 indices): {self.manifold2_indices[:10]}")
        logger.debug(f"Manifold 2 indices sample (second 10 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of manifold 2 indices before shuffling: {self.manifold2_indices.shape}")

        self.manifold2_indices = self.manifold2_indices[data_indices]

        logger.debug(f"Manifold 2 indices sample after shuffling (first 10 indices): {self.manifold2_indices[:10]}")
        logger.debug(f"Manifold 2 indices sample after shuffling (second 10 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of manifold 2 indices after shuffling: {self.manifold2_indices.shape}")

        self.labels = np.concatenate([labels_0_1, labels_1_0], axis=0, dtype=np.float32)

        logger.debug(f"Labels sample (first 10 labels): {self.labels[:10]}")
        logger.debug(f"Labels sample (second 10 labels): {self.labels[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of labels before shuffling: {self.labels.shape}")

        self.labels = self.labels[data_indices]

        logger.debug(f"Labels sample (first 10 labels): {self.labels[:10]}")
        logger.debug(f"Labels sample (second 10 labels): {self.labels[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of labels after shuffling: {self.labels.shape}")

        self.imgs_root_path = img_root_path
        self.model_type = model_type
        self.task_mode = task_mode
        if self.task_mode == "test":
            with open(sem_idxes_path, "r") as sem_idxes_file:
                self.semantic_indices = json.load(sem_idxes_file)

            logger.debug(f"Semantic indices: {self.semantic_indices}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path_1, img_path_2 = self.data[idx]
        assert img_path_1 is not None and img_path_2 is not None, "Image paths cannot be None"
        assert img_path_1.endswith(".jpg") or img_path_1.endswith(".jpeg") or \
               img_path_1.endswith(".JPEG") or img_path_1.endswith(".png"),\
               "Image path must end with .jpg, .jpeg, .JPEG or .png"
        assert img_path_2.endswith(".jpg") or img_path_2.endswith(".jpeg") or \
               img_path_2.endswith(".JPEG") or img_path_2.endswith(".png"),\
               "Image path must end with .jpg, .jpeg, .JPEG or .png"

        label_1, label_2 = self.labels[idx]
        if self.task_mode == "test":
            semantic_label_1 = torch.tensor(int(self.semantic_indices[img_path_1.split("/")[-1].split("_")[0]])).long()
            semantic_label_2 = torch.tensor(int(self.semantic_indices[img_path_2.split("/")[-1].split("_")[0]])).long()
        img_1 = Image.open(os.path.join(self.imgs_root_path, img_path_1)).convert("RGB")
        img_2 = Image.open(os.path.join(self.imgs_root_path, img_path_2)).convert("RGB")
        if self.model_type == "CLIP":
            transform = self.transform_img_clip()
        elif self.model_type == "DINO_V2":
            transform = self.transform_img_dinov2()
        elif self.model_type == "DINO_V3_LVD":
            transform = self.transform_img_lvd()
        elif self.model_type == "DINO_V3_SAT":
            transform = self.transform_img_sat()
        img_tensor_1 = transform(img_1)
        img_tensor_2 = transform(img_2)
        
        label_tensor_1 = torch.tensor(label_1, requires_grad=False).float()
        label_tensor_2 = torch.tensor(label_2, requires_grad=False).float()
        
        if self.task_mode == "test":
            return (img_tensor_1, img_tensor_2), torch.tensor(self.manifold2_indices[idx]).long(),\
                   (label_tensor_1, label_tensor_2), (semantic_label_1, semantic_label_2)
        else:
            return (img_tensor_1, img_tensor_2), torch.tensor(self.manifold2_indices[idx]).long(),\
                   (label_tensor_1, label_tensor_2)

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

    def transform_img_dinov2(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

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


class DFDDFMPredictDataset(Dataset):
    def __init__(self,
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 imgs_path: str,
                 sem_idxes_path: str = None,
                 dfd_idxes_path: str = None):
        """
            params:
                model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model to be used. Options are "CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT".
                imgs_path: str The path to the images.
                sem_idxes_path: str The path to the semantic classes.
                dfd_idxes_path: str The path to the deepfake detection classes.
        """
        assert os.path.exists(imgs_path), f"Image path {imgs_path} does not exist."
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"], f"model_type should be one of ['CLIP', 'DINO_V2', 'DINO_V3_LVD', 'DINO_V3_SAT'], but got {model_type}."
        assert sem_idxes_path is not None and os.path.exists(sem_idxes_path), f"sem_idxes_path {sem_idxes_path} does not exist."
        assert dfd_idxes_path is not None and os.path.exists(dfd_idxes_path), f"dfd_idxes_path {dfd_idxes_path} does not exist."
        
        with open(sem_idxes_path, "r") as sem_idxes_file:
            self.semantic_indices = json.load(sem_idxes_file)
        with open(dfd_idxes_path, "r") as dfd_idxes_file:
            self.dfd_indices = json.load(dfd_idxes_file)
        
        self.data = np.array(os.listdir(imgs_path))
        self.semantic_labels = np.array([int(self.semantic_indices[img_path]) for img_path in self.data], dtype=np.int64)
        self.dfd_labels = np.array([int(self.dfd_indices[img_path]) for img_path in self.data], dtype=np.float32)
        data_indices = np.arange(self.data.shape[0])
        np.random.shuffle(data_indices)
        self.data = self.data[data_indices]
        self.semantic_labels = self.semantic_labels[data_indices]
        self.dfd_labels = self.dfd_labels[data_indices]
        self.imgs_path = imgs_path
        self.model_type = model_type

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        assert img_path is not None, "Image path cannot be None"
        assert img_path.endswith(".jpg") or img_path.endswith(".jpeg") or \
               img_path.endswith(".JPEG") or img_path.endswith(".png") or \
               img_path.endswith(".PNG") or img_path.endswith(".gif"), "Invalid image format"
        
        semantic_label = torch.tensor(self.semantic_labels[idx], requires_grad=False).long()
        dfd_label = torch.tensor([self.dfd_labels[idx]], requires_grad=False).float()
        img = Image.open(os.path.join(self.imgs_path, img_path)).convert("RGB")
        
        if self.model_type == "CLIP":
            transform = self.transform_img_clip()
        elif self.model_type == "DINO_V2":
            transform = self.transform_img_dinov2()
        elif self.model_type == "DINO_V3_LVD":
            transform = self.transform_img_lvd()
        elif self.model_type == "DINO_V3_SAT":
            transform = self.transform_img_sat()
        
        img_tensor = transform(img)

        return img_tensor, semantic_label, dfd_label

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

    def transform_img_dinov2(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

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


class DFDDFMFeaturesDataset(Dataset):
    def __init__(self,
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 imgs_path: str,
                 imgs_order_path: str):
        """
            params:
                model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model to be used. Options are "CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT".
                imgs_path: str The path to the images.
                imgs_order_path: str The path to the images order file.
        """
        assert imgs_path is not None and os.path.exists(imgs_path), f"Image path {imgs_path} does not exist."
        assert imgs_order_path is not None and os.path.exists(imgs_order_path), f"Image order path {imgs_order_path} does not exist."
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"], f"model_type should be one of ['CLIP', 'DINO_V2', 'DINO_V3_LVD', 'DINO_V3_SAT'], but got {model_type}."
        
        with open(imgs_order_path, "r") as imgs_order_file:
            imgs_order = json.load(imgs_order_file)
        self.data = np.array([img_path for img_path in imgs_order])
        self.model_type = model_type
        self.imgs_path = imgs_path
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        assert img_path is not None, "Image path cannot be None"
        assert img_path.endswith(".jpg") or img_path.endswith(".jpeg") or \
               img_path.endswith(".JPEG") or img_path.endswith(".png") or \
               img_path.endswith(".PNG") or img_path.endswith(".gif"), "Invalid image format"

        img = Image.open(os.path.join(self.imgs_path, img_path)).convert("RGB")
        
        if self.model_type == "CLIP":
            transform = self.transform_img_clip()
        elif self.model_type == "DINO_V2":
            transform = self.transform_img_dinov2()
        elif self.model_type == "DINO_V3_LVD":
            transform = self.transform_img_lvd()
        elif self.model_type == "DINO_V3_SAT":
            transform = self.transform_img_sat()
        
        img_tensor = transform(img)

        return img_tensor

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

    def transform_img_dinov2(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

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


class DFDDFMTrainDataModule(LTN.LightningDataModule):
    def __init__(self, 
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 train_dataset_path: str,
                 val_dataset_path: str,
                 test_dataset_path: str,
                 manifolds_paths: Tuple[str, str],
                 manifolds_indices_path: str = None,
                 sem_idxes_path: str = None,
                 batch_size: int = 120,
                 num_workers: int = 2,
                 predict_dataset_path: str = None,
                 feat_only: bool = False,
                 imgs_order_path: str = None,
                 dfd_idxes_path: str = None):
        """
        Initialize the DFDDFMTrainDataModule with the given dataset paths.
        Params:
            model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model for the image transformations.
            train_dataset_path: str The path to the training dataset.
            val_dataset_path: str The path to the validation dataset.
            test_dataset_path: str The path to the test dataset.
            manifolds_paths: Tuple[str, str] The paths to the manifolds images paths.
            manifolds_indices_path: str The path to the mapping of the corresponding manifolds' indices.
            sem_idxes_path: str The path to the semantic indices. Required if test_dataset_path or predict_dataset_path is provided.
            batch_size: int The batch size for the dataloaders.
            num_workers: int The number of workers for the dataloaders.
            predict_dataset_path: str The path to the prediction dataset.
            feat_only: bool If True, only load the features without labels.
            imgs_order_path: str The path to the images order file. Required if feat_only is True.
            dfd_idxes_path: str The path to the deepfake detection indices. Required if predict_dataset_path is provided.
        """
        super(DFDDFMTrainDataModule, self).__init__()

        assert not (train_dataset_path is None and val_dataset_path is None and test_dataset_path \
               is None and predict_dataset_path is None), \
               "At least one of train_dataset_path, val_dataset_path, test_dataset_path, predict_dataset_path must be provided."
        if not train_dataset_path:
            assert train_dataset_path.endswith("TRAIN") or train_dataset_path.endswith("train"), f"Invalid train dataset path {train_dataset_path}"
        if not val_dataset_path:
            assert val_dataset_path.endswith("VAL") or val_dataset_path.endswith("val"), f"Invalid val dataset path {val_dataset_path}"
        if not test_dataset_path:
            assert test_dataset_path.endswith("TEST") or test_dataset_path.endswith("test"), f"Invalid test dataset path {test_dataset_path}"
        if predict_dataset_path is None or predict_dataset_path == "":
            assert manifolds_indices_path is not None and os.path.exists(manifolds_indices_path), f"manifolds_indices_path {manifolds_indices_path} does not exist."
        if predict_dataset_path is not None and predict_dataset_path != "":
            assert feat_only is not None and isinstance(feat_only, bool), f"feat_only should be a boolean value, but got {type(feat_only)}."
            if feat_only:
                assert imgs_order_path is not None and os.path.exists(imgs_order_path), f"imgs_order_path {imgs_order_path} does not exist."

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model_type = model_type
        self.manifolds_paths = manifolds_paths
        if predict_dataset_path is None or predict_dataset_path == "":
            with open(manifolds_indices_path, "r") as mani_idxes_file:
                self.manifolds_indices_correspondence = json.load(mani_idxes_file)
        
        if predict_dataset_path is not None and predict_dataset_path != "":
            self.feat_only = feat_only
            if feat_only:
                self.imgs_order_path = imgs_order_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sem_idxes_path = sem_idxes_path
        self.dfd_idxes_path = dfd_idxes_path
        self.predict_dataset_path = predict_dataset_path

    def setup(self, stage=None):
        if stage == "fit":
            assert self.train_dataset_path is not None, "train_dataset_path must be provided for fit"

            self.train_dataset = DFDDFMDataset(self.model_type,
                                               self.train_dataset_path,
                                               self.manifolds_paths,
                                               "ntest",
                                               self.sem_idxes_path,
                                               self.manifolds_indices_correspondence)
            if self.val_dataset_path is not None:
                self.val_dataset = DFDDFMDataset(self.model_type,
                                                 self.val_dataset_path,
                                                 self.manifolds_paths,
                                                 "ntest",
                                                 self.sem_idxes_path,
                                                 self.manifolds_indices_correspondence)
        # if stage == "validate":
        #     self.val_dataset = DFDDFMDataset(self.model_type, self.val_dataset_path, self.manifolds_paths)
        if stage == "test":
            assert self.test_dataset_path is not None, "test_dataset_path must be provided for test"

            self.test_dataset = DFDDFMDataset(self.model_type,
                                              self.test_dataset_path,
                                              self.manifolds_paths,
                                              "test",
                                              self.sem_idxes_path,
                                              self.manifolds_indices_correspondence)
        if stage == "predict":
            assert self.predict_dataset_path is not None, "predict_dataset_path must be provided for predict"

            if self.feat_only:
                self.predict_dataset = DFDDFMFeaturesDataset(self.model_type,
                                                             self.predict_dataset_path,
                                                             self.imgs_order_path)
            else:
                self.predict_dataset = DFDDFMPredictDataset(self.model_type,
                                                            self.predict_dataset_path,
                                                            self.sem_idxes_path,
                                                            self.dfd_idxes_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True,
                          shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, pin_memory=True,
                          shuffle=False, num_workers=self.num_workers)