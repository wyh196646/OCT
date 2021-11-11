import numpy as np
import cv2


def cut_cp_from_imgs(img_array, volume_size, flip=False, radius=2.5, threshold=100, size=(380, 380)):
    """ Create a CP from a list of OCT slices
    When flip set to True for left eye and False for right eye, 
    the CP will be displayed in an order of TSNIT, from left to right, as the sample image in the review paper.
    
    img_array: a list of 256 OCT slice images with a size of 512 x 992.
    volume_size: the actual size of the OCT volume from the volume info, eg: 6 for a volumne of 6mmx6mm.
    flip: flip the input volume. True for left eye and False for right eye.
    radius: the radius of CP. Do not set radius > volume_size / 2.
    threshold: the acceptable distance threshold between the distance of point to the center and the radius
    size: output size of CP image
    """

    volume = np.stack(img_array, axis=0)
    volume = volume.transpose(0,2,1)
    volume = volume[:, ::2, :]
    if flip:
        volume = np.flip(volume, axis=1)
    y, x = np.mgrid[255:-1:-1, 0:256:1]
    
    dis = (x - 255 / 2) ** 2 + (y - 255 / 2) ** 2
    valid_mask = np.abs(dis - (radius / volume_size * 256) ** 2) < threshold
    theta = np.arctan2(y - 255/2, x - 255/2)
    
    masked_theta = theta[valid_mask]
    masked_volume = volume[valid_mask, :]
    img = masked_volume[masked_theta.argsort()]
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    return img.transpose()
