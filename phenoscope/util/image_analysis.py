from seaborn import histplot
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


def check_grayscale(img: np.ndarray):
    if len(img.shape) == 3:
        return rgb2gray(img)
    elif len(img.shape) == 2:
        return img
    else:
        raise ValueError('Image must be grayscale or RGB')


def plotAx_histogram(
        ax, grayscale_img, bins=256, stat="count"
):
    histplot(grayscale_img.ravel(), ax=ax, stat=stat, bins=bins)


def plot_histogram(grayscale_img):
    fig, ax = plt.subplots()
    histplot(grayscale_img.ravel(), ax=ax)
    return fig, ax


def compare_hist(img_one, img_two, figsize=(12, 8)):
    if len(img_one.shape) == 2:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        histplot(img_one.ravel(), bins=256, ax=axes[0])
        histplot(img_two.ravel(), bins=256, ax=axes[1])
    else:
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=figsize)
        ax = axes.ravel()
        histplot(img_one[:, :, 0], bins=256, ax=ax[0])
        ax[0].set_title("Red 1")
        histplot(img_two[:, :, 0], bins=256, ax=ax[1])
        ax[1].set_title("Red 2")
        histplot(img_one[:, :, 1], bins=256, ax=ax[2])
        ax[2].set_title("Green 1")
        histplot(img_two[:, :, 1], bins=256, ax=ax[3])
        ax[3].set_title("Green 2")
        histplot(img_one[:, :, 2], bins=256, ax=ax[4])
        ax[4].set_title("Blue 1")
        histplot(img_two[:, :, 2], bins=256, ax=ax[5])
        ax[5].set_title("Blue 2")

    return fig, axes


def view_img_info(img, cmap='viridis', stat="count", bins=256, sharey=False, sharex=False, figsize=(14, 8)):
    if len(img.shape) == 2:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        axes[0].imshow(img, cmap=cmap)
        axes[0].grid(False)
        axes[0].set_title("Image")
        histplot(data=img.ravel(),
                 stat=stat, bins=bins, ax=axes[1])
    else:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
        ax = axes.ravel()
        ax[0].imshow(img)
        ax[0].grid(False)
        ax[0].set_title("Image")
        channel = ["Red", "Green", "Blue"]
        ylims = []
        xlims = []
        for idx, _ax in enumerate(ax[1:]):
            histplot(img[:, :, idx].ravel(),
                     stat=stat, bins=bins, ax=_ax)
            _ax.set_title(f"{channel[idx]}")
            ylims.append(_ax.get_ylim()[1])
            xlims.append(_ax.get_xlim()[1])

        if sharey:
            yax = ax[1:].tolist()
            max_yax = yax.pop(ylims.index(max(ylims)))
            for _ax in yax:
                _ax.sharey(max_yax)

        if sharex:
            xax = ax[1:].tolist()
            max_xax = xax.pop(xlims.index(max(xlims)))
            for _ax in xax:
                _ax.sharex(max_xax)

    return fig, axes
