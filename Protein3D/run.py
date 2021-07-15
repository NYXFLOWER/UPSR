#%%
from datasets import *

from equivariant_attention.utils_profiling import *
from dgl.data.utils import save_graphs
import math
from torch import nn, optim
from torch.nn import functional as F

from models import SE3Transformer 
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.callbacks import ModelCheckpoint

# ##################### Hyperpremeter Setting #########################
class ExpSetting(object):
	print_interval = 1

	def __init__(self, distance_cutoff=[3, 3.5], data_address='../data/ProtFunct.pt', batch_size=8, lr=1e-3, num_epochs=2, num_workers=4, num_layers=2, num_degrees=3, num_channels=20, num_nlayers=0, pooling='avg', head=1, div=4, seed=0): 
		self.distance_cutoff = distance_cutoff
		self.data_address = data_address
		self.batch_size = batch_size
		self.lr = lr          		 	  # learning rate
		self.num_epochs = num_epochs          
		self.num_workers = num_workers

		self.num_layers = num_layers      # number of equivariant layer
		self.num_degrees = num_degrees    # number of irreps {0,1,...,num_degrees-1}
		self.num_channels = num_channels  # number of channels in middle layers
		self.num_nlayers = num_nlayers    # number of layers for nonlinearity
		self.pooling = pooling        	  # choose from avg or max
		self.head = head                  # number of attention heads
		self.div = div                    # low dimensional embedding fraction

		self.seed = seed                  # random seed for both numpy and pytorch

		self.n_bounds = len(distance_cutoff) + 1
		
		self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')				 # Automatically choose GPU if available


# TODO: change to multiple class prediction
pred_class_binary = 3


# #####################################################################

class ProtBinary(pl.LightningModule):
	def __init__(self, setting: ExpSetting):
		super().__init__()
		self.setting = setting
		self.model = SE3Transformer(setting.num_layers, len(residue2idx), setting.num_channels, setting.num_nlayers, setting.num_degrees, edge_dim=3, n_bonds=setting.n_bounds, div=setting.div, pooling=setting.pooling, head=setting.head)

	def forward(self, x):
		pass
	# 	"""get model prediction"""
    #     embedding = self.encoder(x)
    #     return embedding

	def _run_step(self, g):
		"""compute forward"""
		z = self.model(g)
		return torch.sigmoid(z)

	# def _shared_eval_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.model(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     acc = FM.accuracy(y_hat, y)
    #     return loss, acc	

	def step(self, batch, batch_idx):
		# print(batch_idx)
		g, y_org, pdb = batch
		
		
		y = torch.zeros(y_org.shape[0])
		y[y_org==pred_class_binary] = 1

		pred = self._run_step(g)

		y = y.type_as(pred)

		l1_loss = torch.sum(torch.abs(pred - y))
		l2_loss = torch.sum((pred - y)**2)
		
		# if use_mean:
		l1_loss /= pred.shape[0]
		l2_loss /= pred.shape[0]		

		loss = l1_loss

		logs = {
			"l1_loss": l1_loss,
			"l2_loss": l2_loss,
		}
		return loss, logs

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"val_{k}": v for k, v in logs.items()})
		return loss

	def test_step(self, batch, batch_idx):
		pass

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), self.setting.lr)
		scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.setting.num_epochs, eta_min=1e-4)
		return [optimizer], [scheduler]

	# # Loss function
	# def task_loss_binary_class(pred, target, use_mean=True):
	# 	l1_loss = torch.sum(torch.abs(pred - target))
	# 	l2_loss = torch.sum((pred - target)**2)
	# 	if use_mean:
	# 		l1_loss /= pred.shape[0]
	# 		l2_loss /= pred.shape[0]

	# 	# rescale_loss = train_dataset.norm2units(l1_loss)
	# 	# rescale_loss = 0
	# 	return l1_loss, l2_loss

# *** load experimental settings ***
setting = ExpSetting()

torch.manual_seed(setting.seed)		# fix seed for random numbers
np.random.seed(setting.seed)

# *** prepare data ***
train_dataset = ProtFunctDataset(setting.data_address, 
	mode='train', 
	if_transform=True, 
	dis_cut=setting.distance_cutoff)

train_loader = DataLoader(train_dataset, 
	batch_size=setting.batch_size, 
	shuffle=False, 
	collate_fn=collate, 
	num_workers=setting.num_workers)

# *** init model ***
autoencoder = ProtBinary(setting)

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='l1_loss',
    dirpath='save/',
    filename='Binary-{pred_class_binary}-{epoch:02d}-{l1_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = pl.Trainer(gpus=1) 		# if you have GPUs
# trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader)






# # *** construct model ***
# model = SE3Transformer(setting.num_layers, train_dataset.atom_feature_size, setting.num_channels, setting.num_nlayers, setting.num_degrees, edge_dim=3, n_bonds=setting.n_bounds, div=setting.div, pooling=setting.pooling, head=setting.head)
# model.to(setting.device)

# # *** Optimizer settings ***
# optimizer = optim.Adam(model.parameters(), lr=setting.lr)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, setting.num_epochs, eta_min=1e-4)



# model.train()

# epoch = 0
# for i, (g, y_org, pdb) in enumerate(train_loader):
# 	# if i == 2:
# 	# 	break
# 	print(pdb)

# 	# TODO: change to multiple class prediction
# 	pred_class_binary = 3
# 	y = torch.zeros(y_org.shape[0])
# 	y[y_org==pred_class_binary] = 1

# 	g, y = g.to(setting.device), y.to(setting.device)

# 	optimizer.zero_grad()

# 	# run model forward and compute loss
# 	pred = model(g)
# 	pred = torch.sigmoid(pred)
# 	l1, l2, rl = task_loss_binary_class(pred, y)
# 	# print(y, pred)

# 	# backprop
# 	l1.backward()
# 	optimizer.step()
# 	# if i % setting.print_interval == 0:
# 	# 	print(f"[{epoch}|{i}] l1 loss: {l1:.5f} rescale loss: {rl:.5f} [units]")
       



# %%
