class Config:
    patch_size = 256
    __conf = {
        "datasets_path": "./Datasets/",
        "images_list": "./images_list.txt",
        "patch_shape_grayscale": (patch_size, patch_size),
        "patch_shape_rgb": (patch_size, patch_size, 3),
        "patch_step": 256
    }

    @staticmethod
    def Get(property):
        return Config.__conf[property]
