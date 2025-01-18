from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
import csv
import gzip
import os
import random
import shutil
import zipfile
from typing import Dict, Generator, List, Optional

from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit, train_test_split, StratifiedShuffleSplit
import numpy as np
import pysmiles
import torch
from sklearn.preprocessing import LabelBinarizer 

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import MergedDataset, XYBaseDataModule
from chebai.preprocessing.datasets.chebi import JCIExtendedTokenData
from chebai.preprocessing.datasets.pubchem import Hazardous

class ClinTox(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "FDA_APPROVED",
        "CT_TOX",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "ClinTox"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 2

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["clintox.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """Returns a list of processed file names."""
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self) -> None:
        """Downloads and extracts the dataset."""
        with NamedTemporaryFile("rb") as gout:
            request.urlretrieve(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                gout.name,
            )
            with gzip.open(gout.name) as gfile:
                with open(os.path.join(self.raw_dir, "clintox.csv"), "wt") as fout:
                    fout.write(gfile.read().decode())

    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(self._load_data_from_file(os.path.join(self.raw_dir, f"clintox.csv")))
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
            print(self.train_split)
            print(type(data))
            print((data[0]))
            print(type(data[0]))
            X = []
            y = []
            for item in data:
                X.append(item['ident'])
                y.append(item['labels'])
            sss = StratifiedShuffleSplit(n_splits=10, test_size=1-self.train_split, random_state=0)
            sss.get_n_splits(np.array(X), np.array(y))
            print(sss)
            train, test = sss.split(X, y)
            print(train)
            exit()
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

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
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

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [
                    bool(int(l)) if l else None for l in (row[k] for k in self.HEADERS)
                ]
                yield dict(features=smiles, labels=labels, ident=i)
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))


class BBBP(XYBaseDataModule):
    """Data module for ClinTox MoleculeNet dataset."""

    HEADERS = [
        "p_np",
    ]

    @property
    def _name(self) -> str:
        """Returns the name of the dataset."""
        return "BBBP"

    @property
    def label_number(self) -> int:
        """Returns the number of labels."""
        return 1

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names."""
        return ["bbbp.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """Returns a list of processed file names."""
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self) -> None:
        
        """Downloads and extracts the dataset."""
        with open(os.path.join(self.raw_dir, "bbbp.csv"), "ab") as dst:
            with request.urlopen(f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",) as src:
                shutil.copyfileobj(src, dst)


    def setup_processed(self) -> None:
        """Processes and splits the dataset."""
        print("Create splits")
        data = list(self._load_data_from_file(os.path.join(self.raw_dir, f"bbbp.csv")))

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

    def setup(self, **kwargs) -> None:
        """Sets up the dataset by downloading and processing if necessary."""
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

    def _load_dict(self, input_file_path: str) -> List[Dict]:
        """Loads data from a CSV file.

        Args:
            input_file_path (str): Path to the CSV file.

        Returns:
            List[Dict]: List of data dictionaries.
        """
        i = 0
        with open(input_file_path, "r") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                i += 1
                smiles = row["smiles"]
                labels = [int(row["p_np"])]
                yield dict(features=smiles, labels=labels, ident=i)
                # yield self.reader.to_data(dict(features=smiles, labels=labels, ident=i))


class BBBPChem(BBBP):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader


class ClinToxChem(ClinTox):
    """Chemical data reader for Tox21MolNet dataset."""

    READER = dr.ChemDataReader