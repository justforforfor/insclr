from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def imread(img_path):
    with open(img_path, "rb") as fr:
        img = Image.open(fr).convert("RGB")
    return img


def imresize(img, size):
    img.thumbnail((size, size), Image.ANTIALIAS)
    return img
