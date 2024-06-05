from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
import csv
import gzip
import os
import random
import shutil
import zipfile

from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import numpy as np
import pysmiles
import torch

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import MergedDataset, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import JCIExtendedTokenData
from chebai.preprocessing.datasets.pubchem import Hazardous


class SolCuration(XYBaseDataModule):
    HEADERS = [
        "logS",
    ]

    @property
    def _name(self):
        return "SolCuration"

    @property
    def label_number(self):
        return 12

    @property
    def raw_file_names(self):
        return ["solCuration.csv"]

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        with NamedTemporaryFile("rb") as gout:
            # start with downloading just one part of the dataset, later add the remaining ones
            request.urlretrieve(
                "https://github.com/Mengjintao/SolCuration/blob/master/cure/esol_cure.csv",
                gout.name,
            )
            with gout.name as gfile:
               with open(os.path.join(self.raw_dir, "solCuration.csv"), "wt") as fout:
                   fout.write(gfile.read())

    def setup_processed(self):
        print("Create splits")
        data = self._load_data_from_file(os.path.join(self.raw_dir, f"solCuration.csv"))
        groups = np.array([d["group"] for d in data])
        if not all(g is None for g in groups):
            split_size = int(len(set(groups)) * self.train_split)
            os.makedirs(self.processed_dir, exist_ok=True)
            splitter = GroupShuffleSplit(train_size=split_size, n_splits=1)

            train_split_index, temp_split_index = next(
                splitter.split(data, groups=groups)
            )

            split_groups = groups[temp_split_index]

            splitter = GroupShuffleSplit(
                train_size=int(len(set(split_groups)) * self.train_split), n_splits=1
            )
            test_split_index, validation_split_index = next(
                splitter.split(temp_split_index, groups=split_groups)
            )
            train_split = [data[i] for i in train_split_index]
            test_split = [
                d
                for d in (data[temp_split_index[i]] for i in test_split_index)
                if d["original"]
            ]
            validation_split = [
                d
                for d in (data[temp_split_index[i]] for i in validation_split_index)
                if d["original"]
            ]
        else:
            train_split, test_split = train_test_split(
                data, train_size=self.train_split, shuffle=True
            )
            test_split, validation_split = train_test_split(
                test_split, train_size=0.5, shuffle=True
            )
        for k, split in [
            ("test", test_split),
            ("train", train_split),
            ("validation", validation_split),
        ]:
            print("transform", k)
            torch.save(
                split,
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    def setup(self, **kwargs):
        if any(
            not os.path.isfile(os.path.join(self.raw_dir, f))
            for f in self.raw_file_names
        ):
            self.download()
        if any(
            not os.path.isfile(os.path.join(self.processed_dir, f))
            for f in self.processed_file_names
        ):
            self.setup_processed()

    def _load_dict(self, input_file_path):
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                smiles = row["smiles"]
                labels = row["logS"]
                # dataset has no mol_id TODO
                yield dict(features=smiles, labels=labels, ident=i) #, ident=row["mol_id"]
                i += 1

class SolubilityCuratedData(SolCuration):
    READER = dr.ChemDataReader
