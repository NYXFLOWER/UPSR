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
# from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_binary_class(class_idx):
    # *** load experimental settings ***
    # class_idx = 0   
    setting = ExpSetting(log_dir=f'log/class_{class_idx}/', batch_size=2) 

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED
    pl.seed_everything(setting.seed, workers=True)

    # *** init model ***
    # # -------- using binary classifier -------- 
    # model = ProtBinary(setting, pred_class_binary=3)
    # -------- using multi-label classifier -------- 
    model = ProtBinaryClass(setting, class_idx=class_idx)

    # continue training
    # model = ProtMultClass.load_from_checkpoint(setting=setting, checkpoint_path='/home/flower/github/Protein3D/Protein3D/save/epoch=09-train_l1_loss=2348.64.ckpt')

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss_epoch',
        dirpath=f'/home/flower/projects/def-laurence/flower/save/ec_binary/class_{class_idx}/',
        # dirpath=f'save/class_{class_idx}/',
        filename='{epoch:02d}-{valid_loss:.4f}',
        save_top_k=5,
        mode='min',
    )

    logger = TensorBoardLogger("tb_log_ec_binary/", name=f"class_{class_idx}", log_graph=True)

    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger, callbacks=[checkpoint_callback]) 		# if you have GPUs
    trainer.fit(model)
    trainer.test(model)

# trainer.validate(model, model.validation_dataloader())
# trainer.validate(model)
# t = trainer.test(model)

for i in range(384):
    print(f"============ MODEL {i:3d} ============")
    train_binary_class(i)



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
