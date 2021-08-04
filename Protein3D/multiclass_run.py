#%%
import os
os.chdir('/home/flower/github/Protein3D/Protein3D')

from models import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation(test_mode=False)

def train_multiclass(setting: ExpSetting):
    pl.seed_everything(setting.seed, workers=True)

    model = ProtMultClass(setting)

    # continue training
    # model = ProtMultClass.load_from_checkpoint(setting=setting, checkpoint_path='loss=2348.64.ckpt')
    cp_dir = f'/home/flower/projects/def-laurence/flower/save/ec_multiclass_t/{setting.hyperparameter}/'

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss_epoch',
        dirpath=cp_dir,
        # dirpath=f'save/class_{class_idx}/',
        filename='{epoch:02d}-{valid_loss_epoch:.4f}',
        save_top_k=3,
        mode='min',
    )

    logger = TensorBoardLogger("tb_log_ec_multiclass/", name=setting.hyperparameter, log_graph=True)
    # trainer = pl.Trainer(gpus=1, max_epochs=20, logger=logger, callbacks=[checkpoint_callback]) 
    trainer = pl.Trainer(max_epochs=20, logger=logger, callbacks=[checkpoint_callback]) 

    # print(f'============== epoch {epoch} ===============')
    trainer.fit(model)
    trainer.test(model)

distance_cutoff = [3, 3.5]
num_layer = 2
num_degrees = 3
num_channels = 20
head = 1
# use_classes = [357, 351, 23, 215, 190, 162, 62, 195, 321, 369] # 10 most commons
use_classes = [321, 369]
batch_size = 2
decoder_mid_dim = 60


# for decoder_mid_dim in [16, 32, 48, 64]:
# for head in [2, 3, 4, 5, 6]:
hyperparameter = f"{distance_cutoff[1]}-{num_layer}-{num_degrees}-{num_channels}-{head}-{batch_size}-{decoder_mid_dim}"
log_dir = f'log_mc_2tt/{hyperparameter}'

setting = ExpSetting(log_dir=log_dir, hyperparameter=hyperparameter, distance_cutoff=distance_cutoff, num_layers=num_layer, num_degrees=num_degrees, num_channels=num_channels, head=head, use_classes=use_classes, batch_size=batch_size, decoder_mid_dim=decoder_mid_dim)

print("Hyperparameter: ", setting.hyperparameter)
train_multiclass(setting)
# %%
