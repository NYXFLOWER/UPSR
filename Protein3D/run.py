#%%
import os
os.chdir('/home/flower/github/Protein3D/Protein3D')

from equivariant_attention.utils_profiling import *
from dgl.data.utils import save_graphs
import math
from torch import nn, optim
from torch.nn import functional as F

from models import *

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# *** load experimental settings ***
setting = ExpSetting(log_file='log')    

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED
pl.seed_everything(setting.seed, workers=True)

# *** init model ***
# # -------- using binary classifier -------- 
# model = ProtBinary(setting, pred_class_binary=3)
# -------- using multi-label classifier -------- 
model = ProtMultClass(setting)

# continue training
# model = ProtMultClass.load_from_checkpoint(setting=setting, checkpoint_path='/home/flower/github/Protein3D/Protein3D/save/epoch=09-train_l1_loss=2348.64.ckpt')

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='train_l1_loss',
    dirpath='save/',
    filename='{epoch:02d}-{train_l1_loss:.2f}',
    save_top_k=5,
    mode='min',
)

logger = TensorBoardLogger("lightning_logs", name="test1", log_graph=True)
# trainer = pl.Trainer(max_epochs=3, logger=logger, callbacks=[checkpoint_callback]) 
trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger, callbacks=[checkpoint_callback]) 		# if you have GPUs
trainer.fit(model)
# trainer.validate(model, model.validation_dataloader())
# trainer.validate(model)
# t = trainer.test(model)

# %%
# test block

# for i, data in enumerate(train_loader):
# 	print(i)

#%%
# g, y_org, pdb = data
# y1 = model.model[0](g)
# y2 = model.model[1](y1)
# pred = model.model[2](y2)


# # y_org = torch.tensor([1, 4])
# y = F.one_hot(y_org, num_classes=pred.shape[1])

# pos_pred = pred * y

# l1_loss = torch.sum(torch.abs(pred - y))  + torch.sum(torch.abs(pos_pred - y)) * pred.shape[1]
# l2_loss = torch.sum((pred - y)**2) + torch.sum((pos_pred - y)**2) * pred.shape[1]

# %%
