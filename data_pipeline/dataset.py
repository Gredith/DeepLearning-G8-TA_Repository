"""PyTorch dataset for precomputed face crops. Implementation pending."""

from torch.utils.data import Dataset


class FaceCropDataset(Dataset):
    """Return image, binary label, video ID, and manipulation metadata."""

    def __init__(self, manifest, split: str, transform=None):
        self.manifest = manifest
        self.split = split
        self.transform = transform
        raise NotImplementedError("Load and validate the crop manifest according to E2 T1.")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
