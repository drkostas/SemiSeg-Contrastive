import json

from data.cityscapes_loader import cityscapesLoader
from data.minifrance_loader import minifranceLoader
from data.voc_dataset import VOCDataSet

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "minifrance": minifranceLoader,
        "pascal_voc": VOCDataSet
    }[name]

def get_data_path(name):  # TODO
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '../data/CityScapes/'

    if name == 'gta5':
        return '../data/GTA5/'
    if name == 'pascal_voc':
        return '../data/VOC2012/'
