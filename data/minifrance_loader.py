"""
Code taken from https://github.com/WilhelmT/ClassMix
Slightly modified
"""

import os
import torch
import scipy.misc as m
from torch.utils import data
from data.city_utils import recursive_glob
from data.augmentations import *


class minifranceLoader(data.Dataset):
    """minifranceLoader
    https://ieee-dataport.org/open-access/minifrance
    Data is used for the DFC2022 competition
    https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=(512, 1024),
            img_norm=False,
            augmentations=None,
            return_id=False,
            pretraining='COCO',
            city='Nice'
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.pretraining = pretraining
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 14
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.files = {}
        self.images_base = os.path.join(self.root, city, "BDORTHO")
        # self.annotations_base = os.path.join(self.root, city, "UrbanAtlas_transformed")
        self.annotations_base = os.path.join(self.root, city, "UrbanAtlas")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".tif")
        self.void_classes = [0]
        self.valid_classes = [1, 2, 3, 4, 5, 6, 9, 10, 14]
        self.class_names = [
            "unlabelled",
            "urban_fabric",
            "industrial_commercial_public_military_private_and_transport_units",
            "mine_dump_and_construction_sites",
            "artificial_non_agricultural_vegetated_areas",
            "arable_land",
            "permanent_crops",
            "pastures",
            "complex_and_mixed_cultivation_patterns",
            "orchards_at_the_fringe_of_urban_classes",
            "forests",
            "herbaceous_vegetation_associations",
            "open_spaces_with_little_or_no_vegetation",
            "wetlands",
            "water",
            "clouds_and_shadows",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        self.return_id = return_id

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        try:
            img_path = self.files[self.split][index].rstrip()
            lbl_path = img_path.replace('BDORTHO', 'UrbanAtlas')\
                               .replace('.tif', '_UA2012.tif')
        except Exception as e:
            print("Index: ", index)
            print("Len files: ", len(self.files[self.split]))
            raise e

        try:
            img = m.imread(img_path)
            img = np.array(img, dtype=np.uint8)

            lbl = m.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)
            try:
                if self.augmentations is not None:
                    img, lbl = self.augmentations(img, lbl)
                if self.is_transform:
                    img, lbl = self.transform(img, lbl)
            except Exception as e:
                print("Img Path: ", img_path)
                print("Lbl Path: ", lbl_path)
                print("img: ", img.shape)
                print("lbl: ", lbl.shape)
                raise e

            img_name = img_path.split('/')[-1]
            if self.return_id:
                return img, lbl, img_name, img_name, index
            return img, lbl, img_path, lbl_path, img_name
        except:
            print(img_path)
            self.files[self.split].pop(index)
            return self.__getitem__(index - 1)

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        if self.pretraining == 'COCO':
            img = img[:, :, ::-1]
        img = img.astype(np.float64)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
