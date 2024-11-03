import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_blobs(img, blobs, ax=None, set_axis=False, grayscale=False):
    with plt.ioff():
        if ax is None:
            blobs_fig, blobs_ax = plt.subplots()
        else:
            blobs_ax = ax

        if grayscale is True:
            blobs_ax.imshow(img, cmap='gray')
        else:
            blobs_ax.imshow(img)

        for idx, rows in blobs.iterrows():
            c = plt.Circle((rows['x'], rows['y']), rows['radius'], color='r', fill=False)
            blobs_ax.add_patch(c)
        blobs_ax.grid(False)

        if set_axis is False:
            blobs_ax.set_axis_off()

        if ax is None:
            return (blobs_fig, blobs_ax)
        else:
            return (blobs_ax)


def plotAx_blobs(img, blobs, ax: plt.Axes, grayscale=False):
    with plt.ioff():
        if grayscale is True:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        for idx, rows in blobs.iterrows():
            c = plt.Circle((rows['x'], rows['y']), rows['radius'], color='r', fill=False)
            ax.add_patch(c)
        ax.grid(False)
    return ax


def plot_blobs_by_label(img, blobs, ax=None, set_axis=False, grayscale=False):
    with plt.ioff():
        if ax is None:
            blobs_fig, blobs_ax = plt.subplots()
        else:
            blobs_ax = ax

        if grayscale is True:
            blobs_ax.imshow(img, cmap='gray')
        else:
            blobs_ax.imshow(img)

        blobs_ax.imshow(img)
        for idx, rows in blobs.iterrows():
            c = plt.Circle((rows['x'], rows['y']), rows['radius'], color=rows['color'], fill=False)
            blobs_ax.add_patch(c)
        blobs_ax.grid(False)

        if set_axis is False:
            blobs_ax.set_axis_off()

        if ax is None:
            return (blobs_fig, blobs_ax)
        else:
            return (blobs_ax)


def plot_plate_rows(img, blobs_class, ax=None, set_axis=False, grayscale=False):
    cnames = list(mcolors.TABLEAU_COLORS)
    for i in range(10):  # Support for 1024 Rows
        cnames = cnames + cnames

    with plt.ioff():
        if ax is None:
            blobs_fig, blobs_ax = plt.subplots()
        else:
            blobs_ax = ax

        if grayscale is True:
            blobs_ax.imshow(img, cmap='gray')
        else:
            blobs_ax.imshow(img)

        blobs_ax.imshow(img)
        for set_idx, row_set in enumerate(blobs_class.gridrows):
            for idx, table_rows in row_set.iterrows():
                c = plt.Circle((table_rows['x'], table_rows['y']), table_rows['radius'], color=cnames[set_idx], fill=False)
                blobs_ax.add_patch(c)
            blobs_ax.grid(False)

        if set_axis is False:
            blobs_ax.set_axis_off()

        if ax is None:
            return blobs_fig, blobs_ax
        else:
            return blobs_ax


def plot_plate_cols(img, blobs_class, ax=None, set_axis=False, grayscale=False):
    cnames = list(mcolors.TABLEAU_COLORS)
    for i in range(10):  # Total 1024 Possible Cols
        cnames = cnames + cnames

    with plt.ioff():
        if ax is None:
            blobs_fig, blobs_ax = plt.subplots()
        else:
            blobs_ax = ax

        if grayscale is True:
            blobs_ax.imshow(img, cmap='gray')
        else:
            blobs_ax.imshow(img)

        blobs_ax.imshow(img)
        for set_idx, col_set in enumerate(blobs_class.gridcols):
            for idx, table_cols in col_set.iterrows():
                c = plt.Circle((table_cols['x'], table_cols['y']), table_cols['radius'], color=cnames[set_idx], fill=False)
                blobs_ax.add_patch(c)
            blobs_ax.grid(False)

        if set_axis is False:
            blobs_ax.set_axis_off()

        if ax is None:
            return blobs_fig, blobs_ax
        else:
            return blobs_ax
