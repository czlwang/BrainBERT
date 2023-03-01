import matplotlib.pyplot as plt
import numpy as np

def plot_tensorboard_line(wav, title=None):
    #needs to be [batch, height, width, channels]
    #spec = spec.transpose(0,1).unsqueeze(0)
    #return spec

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.plot(wav)
    plt.xlabel("time")
    plt.ylabel("voltage")
    if title:
        plt.title(title)
    plt.tight_layout()

    fig.canvas.draw()
    data = plot_to_tensorboard(fig)
    plt.close()
    return data

def plot_tensorboard_spectrogram(spec):
    #needs to be [batch, height, width, channels]
    #spec = spec.transpose(0,1).unsqueeze(0)
    #return spec

    spec = spec.transpose(1, 0)
    spec = spec.detach().cpu()
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = plot_to_tensorboard(fig)
    plt.close()
    return data

def plot_to_tensorboard(fig):
    """
    From https://martin-mundt.com/tensorboard-figures/
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.transpose(img, axes=[0,2,1])

    # Add figure in numpy "image" to TensorBoard writer
    plt.close(fig)
    return img

