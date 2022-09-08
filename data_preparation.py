import numpy as np
import config


def add_zero_padding(image, mask, format_NHWC=False):
    """
    Patchify can't handle under-sized patches. Without padding dataset is not fully utilised
    """
    w = 2 if format_NHWC else 0
    h = 3 if format_NHWC else 1
    h_pad = config.PATCH_SIZE - (image.shape[w] % config.PATCH_SIZE)
    v_pad = config.PATCH_SIZE - (image.shape[h] % config.PATCH_SIZE)

    if(h_pad != 0 or v_pad != 0):
        if(format_NHWC):
            img_pad = [(0, 0), (0, 0), (0, h_pad), (0, v_pad)]
            mask_pad = [(0, 0), (0, h_pad), (0, v_pad)]
        else:
            img_pad = [(0, h_pad), (0, v_pad), (0, 0)]
            mask_pad = [(0, h_pad), (0, v_pad)]
        image = np.pad(image, img_pad, mode='constant', constant_values=0)
        mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)
    return image, mask
