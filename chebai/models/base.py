import itertools
import logging
import os
import sys

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.metrics import F1, MeanSquaredError
from pytorch_lightning.tuner.tuning import Tuner
from sklearn.metrics import f1_score
from torch import nn
import pytorch_lightning as pl
import torch
import tqdm

from chebai.preprocessing.datasets import XYBaseDataModule

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class JCIBaseNet(pl.LightningModule):
    NAME = None

    def __init__(self, **kwargs):
        super().__init__()
        weights = kwargs.get("weights", None)
        if weights is not None:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.loss = nn.BCEWithLogitsLoss()
        self.f1 = F1(threshold=kwargs.get("threshold", 0.5), multilabel=True)
        self.mse = MeanSquaredError()
        self.lr = kwargs.get("lr", 1e-4)

        self.save_hyperparameters()

    def _execute(self, batch, batch_idx):
        pred = self(batch)
        labels = batch.y.float()
        loss = self.loss(pred, labels)
        f1 = self.f1(target=labels.int(), preds=torch.sigmoid(pred))
        mse = self.mse(labels, torch.sigmoid(pred))
        return loss, f1, mse

    def training_step(self, *args, **kwargs):
        loss, f1, mse = self._execute(*args, **kwargs)
        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1",
            f1.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_mse",
            mse.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1, mse = self._execute(*args, **kwargs)
            self.log(
                "val_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1",
                f1.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_mse",
                mse.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def test_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1, mse = self._execute(*args, **kwargs)
            self.log(
                "test_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_f1",
                f1.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_mse",
                mse.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @classmethod
    def test(cls, data, name, checkpoint_path):
        data.prepare_data()
        data.setup()
        name += "__" + "_".join(data.full_identifier)
        test_data = data.test_dataloader()

        # Calculate weights per class

        model = cls.load_from_checkpoint(checkpoint_path)

        trainer = pl.Trainer(
            replace_sampler_ddp=False,
        )

        test = trainer.test(model, test_data)
        print(test)

    @classmethod
    def run(
        cls,
        data,
        name,
        model_args: list = None,
        model_kwargs: dict = None,
        weighted=False,
    ):
        if model_args is None:
            model_args = []
        if model_kwargs is None:
            model_kwargs = {}
        data.prepare_data()
        data.setup()
        name += "__" + "_".join(data.full_identifier)
        train_data = data.train_dataloader()
        val_data = data.val_dataloader()

        if weighted:
            weights = model_kwargs.get("weights")
            if weights is None:
                weights = 1 + torch.sum(
                    torch.cat([data.y for data in train_data]).float(), dim=0
                )
                weights = torch.mean(weights) / weights
                name += "__weighted"
            model_kwargs["weights"] = weights
        else:
            try:
                model_kwargs.pop("weights")
            except KeyError:
                pass

        if torch.cuda.is_available():
            trainer_kwargs = dict(gpus=-1, accelerator="ddp")
        else:
            trainer_kwargs = dict(gpus=0)

        tb_logger = pl_loggers.TensorBoardLogger("logs/", name=name)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            filename="{epoch}-{step}-{val_loss:.7f}",
            save_top_k=5,
            save_last=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        # Calculate weights per class

        net = cls(*model_args, **model_kwargs)

        # Early stopping seems to be bugged right now with ddp accelerator :(
        es = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.00,
            verbose=False,
        )

        trainer = pl.Trainer(
            logger=tb_logger,
            max_epochs=model_kwargs.get("epochs", 100),
            callbacks=[checkpoint_callback],
            replace_sampler_ddp=False,
            **trainer_kwargs
        )
        trainer.fit(net, train_data, val_dataloaders=val_data)

    @classmethod
    def pred(cls, dataset: XYBaseDataModule, checkpoint_path, data_path, batch_size=10):
        model = cls.load_from_checkpoint(checkpoint_path)
        with torch.no_grad():
            lines = dataset._get_data_size(data_path)
            iterator = dataset._load_tuples(data_path)
            for i, (smiles, labels) in enumerate(
                tqdm.tqdm(iterator, total=1 + (lines // batch_size))
            ):
                chunk = list(
                    itertools.chain(
                        [(smiles, labels)], itertools.islice(iterator, batch_size - 1)
                    )
                )
                d = dataset.reader.collater(
                    [dataset.reader.to_data(row) for row in chunk]
                )
                pred = torch.sigmoid(model.predict_step(d, i))
                for ((smiles, labels), predicted) in zip(chunk, pred):
                    yield smiles, labels.tolist() if labels is not None else None, (
                        predicted.cpu().numpy().tolist()
                    )
