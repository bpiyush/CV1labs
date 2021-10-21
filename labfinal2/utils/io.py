"""I/O utils"""
import pickle


def decode_string(string, to="utf8"):
    return string.decode(to)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict