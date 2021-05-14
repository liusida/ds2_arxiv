import numpy as np
import cv2
import wandb

def save_pic(matrix, indices, title=""):
    """ save an image and the indices, upload to wandb """
    im = np.array(matrix / matrix.max() * 255, dtype = np.uint8)
    im = 255-im

    # add borders to the image if needed
    border_width = 2
    a = np.zeros(shape=[im.shape[0], border_width], dtype=np.uint8)
    im = np.concatenate([a,im,a], axis=1)
    b = np.zeros(shape=[border_width, border_width+im.shape[0]+border_width ], dtype=np.uint8)
    im = np.concatenate([b,im,b], axis=0)

    cv2.imwrite(f"{title}.png", im)
    wandb.save(f"{title}.png")
    np.save(f"{title}_indicies.npy", indices)
    wandb.save(f"{title}_indicies.npy")
