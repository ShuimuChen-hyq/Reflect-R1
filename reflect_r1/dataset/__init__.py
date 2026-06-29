from .lazy_dataset import LazyVLDataset

def get_dataset_class(dataset_type):
    if dataset_type == "lazy_dataset":
        return LazyVLDataset
    raise ValueError(f"Invalid dataset type: {dataset_type}")
