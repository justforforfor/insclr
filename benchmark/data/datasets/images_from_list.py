from torch.utils.data import Dataset
from .utils import imread, imresize


class ImagesFromList(Dataset):
    def __init__(self, img_paths, bboxes=None, img_size=None, transform=None):
        self.img_paths = img_paths
        self.bboxes = bboxes
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        full_size = max(img.size)

        # crop region of interest
        if self.bboxes is not None:
            img = img.crop(self.bboxes[idx])
        # resize
        if self.img_size is not None:
            img = imresize(img, self.img_size * (max(img.size) / full_size))
        
        input = self.transform(img) if self.transform is not None else img
        return {"input": input}
    
    def __len__(self):
        return len(self.img_paths)
