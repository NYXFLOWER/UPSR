from models import *
from utils import *
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

# setting = ExpSetting(
#     distance_cutoff=[3, 3.5], 
#     num_layers=int(sys.argv[1]),
#     num_degrees=int(sys.argv[2]), 
#     seed=1111,
#     num_channels=int(sys.argv[3]), 
# 	num_workers=0,
#     head=int(sys.argv[4]), 
#     # use_classes = [i for i in range(50)],
#     batch_size=128,
#     lr=1e-3,
# 	num_epochs=50
#     )

setting = ExpSetting(
    distance_cutoff=[3, 3.5], 
    num_layers=2,
    num_degrees=3, 
    seed=1111,
    num_channels=4, 
	num_workers=0,
    head=1, 
    # use_classes = [i for i in range(50)],
    batch_size=64,
    lr=1e-3,
	num_epochs=50
    )

# fix seed
seed_all(setting.seed)

# set model name
name = f"ec{len(setting.use_classes) if setting.use_classes else 'all'}_batch{setting.batch_size*4}_nl{setting.num_layers}_nd{setting.num_degrees}_nc{setting.num_channels}_nh{setting.head}"
print(f'file name: {name}')

# load data
train_loader, valid_loader, test_loader = load_data_ie(setting)

# Init model
model = ProtMultClassLitModel(setting)

# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(
	dirpath=f'/home/flower/github/Protein3D/Protein3D/lightning_logs/pl_checkpoint/{name}/',
	filename='c1-{epoch}-{valid_loss_epoch:.2f}-{accuracy:.2f}',
	monitor="valid_loss_epoch",
	save_top_k=5, 
	mode='min')

# use this trainer when multiple GPUs are available
# trainer = pl.Trainer(
# 	gpus=4, 
# 	max_epochs=setting.num_epochs, 
# 	accelerator="ddp", 
# 	plugins=DDPPlugin(find_unused_parameters=False), 
# 	check_val_every_n_epoch=1, 
# 	callbacks=[checkpoint_callback])

# use this trainier when only one CPU is available
trainer = pl.Trainer()

# fit the model
trainer.fit(model, train_loader, valid_loader)

# load a trained model
# model = ProtMultClassLitModel.load_from_checkpoint(setting=setting, checkpoint_path='/home/flower/github/Protein3D/Protein3D/lightning_logs/pl_checkpoint/ecall_batch512_nl2_nd3_nc4_nh1/epoch=150-valid_loss_epoch=5.33-other_metric=0.00.ckpt')