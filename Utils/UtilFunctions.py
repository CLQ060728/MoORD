from torchvision import transforms

class ConfigDict(dict):
    """
    A dictionary that allows both dot notation and bracket access.
    Fully compatible with normal dict behavior.
    """
    def __getattr__(self, name):
        try:
            value = self[name]
            # Recursively wrap nested dicts
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")


def transform_img_clip(resize_size: int = 256):
    resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                interpolation=transforms.InterpolationMode.BICUBIC)
    center_crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose([resize, center_crop, to_tensor, normalize])


def transform_img_dinov2(resize_size: int = 256):
    resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                interpolation=transforms.InterpolationMode.BICUBIC)
    center_crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return transforms.Compose([resize, center_crop, to_tensor, normalize])


def transform_img_lvd(resize_size: int = 256):
    resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                interpolation=transforms.InterpolationMode.BICUBIC)
    center_crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([resize, center_crop, to_tensor, normalize])


def transform_img_sat(resize_size: int = 256):
    resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                interpolation=transforms.InterpolationMode.BICUBIC)
    center_crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return transforms.Compose([resize, center_crop, to_tensor, normalize])
