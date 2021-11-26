import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Plot the data
def plot(x, y, y_pred, centroids, kmeans):
    colors = ["red", "blue", "green"]
    fig, ax = plt.subplots()
    # Decision Border
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, kmeans, xx, yy, cmap=cmap_light, alpha=0.8)

    plt.scatter(x[:,0], x[:,1], c=y_pred)
    plt.scatter(centroids[:,0], centroids[:,1], color="red", s=250, marker="*")
    plt.show()
