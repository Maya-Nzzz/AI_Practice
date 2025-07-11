import torchvision.transforms as T
from PIL import Image
import os

class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        """Добавить аугментацию в пайплайн."""
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        """Удалить аугментацию из пайплайна по имени."""
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        """Применить все аугментации к изображению."""
        transform_list = list(self.augmentations.values())
        composed = T.Compose(transform_list)
        return composed(image)

    def get_augmentations(self):
        """Вернуть список имен аугментаций."""
        return list(self.augmentations.keys())

# Пример конфигураций
def get_light_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("Resize", T.Resize((256, 256)))
    pipeline.add_augmentation("HorizontalFlip", T.RandomHorizontalFlip(p=0.3))
    return pipeline

def get_medium_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("Resize", T.Resize((256, 256)))
    pipeline.add_augmentation("HorizontalFlip", T.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation("Rotation", T.RandomRotation(20))
    return pipeline

def get_heavy_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("Resize", T.Resize((256, 256)))
    pipeline.add_augmentation("HorizontalFlip", T.RandomHorizontalFlip(p=0.7))
    pipeline.add_augmentation("Rotation", T.RandomRotation(45))
    pipeline.add_augmentation("ColorJitter", T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5))
    return pipeline


def apply_and_save(pipeline, images_dir, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for idx, img_name in enumerate(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        aug_image = pipeline.apply(image)

        save_path = os.path.join(save_dir, f"{prefix}_{idx}.jpg")
        aug_image.save(save_path)


train_dir = "data/для ex_4"
save_dir_light = "results/ex_4/train_aug_light"
save_dir_medium = "results/ex_4/train_aug_medium"
save_dir_heavy = "results/ex_4/train_aug_heavy"

# Создаем пайплайны
light_pipeline = get_light_pipeline()
medium_pipeline = get_medium_pipeline()
heavy_pipeline = get_heavy_pipeline()

# Применяем и сохраняем
apply_and_save(light_pipeline, train_dir, save_dir_light, "light")
apply_and_save(medium_pipeline, train_dir, save_dir_medium, "medium")
apply_and_save(heavy_pipeline, train_dir, save_dir_heavy, "heavy")
