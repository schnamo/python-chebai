import random
from tempfile import TemporaryDirectory
import logging

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraModel,
    PretrainedConfig,
)
import torch

from chebai.models.base import JCIBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ElectraPre(JCIBaseNet):
    NAME = "ElectraPre"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = kwargs["config"]
        self.generator_config = ElectraConfig(**config["generator"])
        self.generator = ElectraForMaskedLM(self.generator_config)
        self.generator_head = torch.nn.Linear(510, 128)
        self.discriminator_config = ElectraConfig(**config["discriminator"])
        self.discriminator = ElectraForPreTraining(self.discriminator_config)
        self.replace_p = 0.1

    def forward(self, data):
        self.batch_size = data.x.shape[0]
        x = torch.clone(data.x)
        gen_tar = []
        dis_tar = []
        for i in range(x.shape[0]):
            j = random.randint(0, x.shape[1]-1)
            t = x[i,j]
            x[i,j] = 0
            gen_tar.append(t)
            dis_tar.append(j)
        gen_out = torch.max(torch.sum(self.generator(x).logits,dim=1), dim=-1)[1]
        with torch.no_grad():
            xc = x.clone()
            for i in range(x.shape[0]):
                xc[i,dis_tar[i]] = gen_out[i]
            replaced_by_different = torch.ne(x, xc)
        disc_out = self.discriminator(xc)
        return (self.generator.electra.embeddings(gen_out.unsqueeze(-1)), disc_out.logits), (self.generator.electra.embeddings(torch.tensor(gen_tar, device=self.device).unsqueeze(-1)), replaced_by_different.float())

    def _get_prediction_and_labels(self, batch, output):
        return output[0][1], output[1][1]


class ElectraPreLoss:

    def __init__(self):
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def __call__(self, target, _):
        t, p = target
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p

        return self.mse(gen_tar, gen_pred) + self.bce(disc_tar, disc_pred)


class Electra(JCIBaseNet):
    NAME = "Electra"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ElectraConfig(**kwargs["config"], output_attentions=True)

        if "pretrained_checkpoint" in kwargs:
            elpre = ElectraPre.load_from_checkpoint(kwargs["pretrained_checkpoint"])
            with TemporaryDirectory() as td:
                elpre.electra.save_pretrained(td)
                self.electra = ElectraModel.from_pretrained(td, config=self.config)
                in_d = elpre.config.hidden_size
        else:
            self.electra = ElectraModel(config=self.config)
            in_d = self.config.hidden_size

        self.output = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, 500),
        )

    def forward(self, data):
        electra = self.electra(data.x)
        d = torch.sum(electra.last_hidden_state, dim=1)
        return self.output(d)
