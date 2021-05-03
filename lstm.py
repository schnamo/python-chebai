import os
from torch import nn
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import MeanSquaredError
from data import JCIExtendedData, JCIData
import logging
import sys

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

class ChemLSTM(pl.LightningModule):

    def __init__(self, in_d, out_d):
        super().__init__()
        self.lstm = nn.LSTM(100, 300, batch_first=True)
        self.embedding = nn.Embedding(800, 100)
        self.output = nn.Sequential(nn.Linear(300, 1000), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1000, 500))
        self.loss = nn.BCEWithLogitsLoss()
        self.f1 = F1(500, threshold=0.5)
        self.mse = MeanSquaredError()


    def _execute(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y.float())
        f1 = self.f1(y, pred)
        return loss, f1, self.mse(y.float(), pred)

    def training_step(self, *args, **kwargs):
        loss, f1, mse = self._execute(*args, **kwargs)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', mse.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1, mse = self._execute(*args, **kwargs)
            self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', f1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mse', mse.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[1][0]
        x = self.output(x)
        return x.squeeze(0)


def run_lstm(batch_size):
    data = JCIData(batch_size=batch_size)
    data.prepare_data()
    data.setup()
    train_data = data.train_dataloader(num_workers=10)
    val_data = data.val_dataloader()
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = ChemLSTM(100, 500)
    tb_logger = pl_loggers.CSVLogger('logs/')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{step}-{val_loss:.7f}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback], max_epochs=100, replace_sampler_ddp=False, **trainer_kwargs)
    trainer.fit(net, train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run_lstm(int(sys.argv[1]))
