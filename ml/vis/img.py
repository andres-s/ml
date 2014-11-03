from matplotlib.pyplot as plt


def display_imgs(imgs, n_rows, n_cols, labels=None, imshow_options=None):
    """Display images in grid.

    imgs is a list of images matrices or a 3+ dimensional numpy ndarray with
    images indexed by first axis.

    imshow_options a dict of keyword args that are passed directly to 
    plt.imshow.
    """
    
    for idx in range(n_rows * n_cols):
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if labels is not None:
            ax.set_title(str(labels[idx]))
        if type(imgs) == list:
            img = imgs[idx]
        else:  # assume numpy array w/ images indexed by first axis
            img = imgs[idx, ...]
        if imshow_options is not None:
            plt.imshow(imgs[idx], **imshow_options)
        elif img.ndim == 2:
            plt.imshow(imgs[idx], cmap='Greys')
        else:
            plt.imshow(imgs[idx])
            
    plt.tight_layout()
    plt.show()