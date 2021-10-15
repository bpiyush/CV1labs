"""Defines constants used throughout the project."""
import numpy as np

# define index to class mapping
path = "./data/stl10_binary/class_names.txt"
classes = open(path, "rb").read().decode("utf-8").split("\n")[:-1]
idx_to_class = {k + 1: v for (k, v) in enumerate(classes)}

# classes to be used for classification (copied from stl)
relevant_classes = np.array([1, 2, 9, 7, 3])
