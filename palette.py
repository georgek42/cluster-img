#!/usr/bin/env python3

from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import numpy as np
import math
import fire

def load_image(filename, dim):
    image = Image.open(filename)
    downscaled = ImageOps.fit(image, (dim, dim))
    return downscaled.getdata()

def normalize(colors):
    # Square the values, because humans perceive color on log-scale
    return np.array([list(map(lambda x: x**2, color)) for color in colors])

def denormalize(colors):
    return [list(map(lambda x: math.sqrt(x), val)) for val in colors]

def find_clusters(values, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(values)
    squared = kmeans.cluster_centers_
    return denormalize(squared)

def make_hashval(r, g, b):
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

class CLI(object):
    def log(self, message):
        if self.verbose:
            print(message)

    def extract(self, filename, num=16, dim=128, verbose=False):
        self.verbose = verbose
        values = load_image(filename, dim)
        self.log('Loaded image')
        normalized = normalize(values)
        self.log('Normalized values')
        centers = find_clusters(normalized, num)
        self.log('Found clusters')
        print([make_hashval(value[0], value[1], value[2]) for value in centers])

if __name__ == '__main__':
    fire.Fire(CLI)