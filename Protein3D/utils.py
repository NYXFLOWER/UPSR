import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_log(log_dir, mode='epoch'):
	assert mode in {'epoch', 'step'}
	if mode == 'epoch':
		df = pd.read_csv(os.path.join(log_dir, f'{mode}.txt'), usecols=range(1, 7), skiprows=1, header=None).values
		data = np.concatenate([df[1:, :3], df[:-1, 3:]], axis=1)
		df = pd.DataFrame(data)
		df.columns = ['loss_valid', 'acc_valid' , 'auroc_valid', 'loss_train', 'acc_train', 'auroc_train']
	else:
		df = pd.read_csv(os.path.join(log_dir, f'{mode}.txt'), usecols=[1])
	return df

def get_traces_by_column(df):
	n = df.shape[0]
	names = df.columns
	values = df.values[:n]
	x = list(range(n))

	trace_list = []
	for c, name in enumerate(names):
		y = values[:, c]
		trace_list.append(
			go.Scatter(x=x, y=y, name=name, mode="lines")
		)

	return trace_list

# def plot_epoch_log(class_name):
#     n_row, n_col = 1, 3
#     fig = make_subplots(rows=n_row, cols=n_col, start_cell="top-left", subplot_titles=("Loss","Accuracy", "AUROC"))
#     use_cols = [[f'{i}_valid', f'{i}_train'] for i in ['loss', 'acc', 'auroc']]

#     for i, use_col in enumerate(use_cols):
#         for trace in get_traces_by_column(epoch_log_dict_by_class[class_name][use_col]):
#             fig.add_trace(trace, row=int(i/n_col)+1, col=i%n_col+1)

#     fig.update_yaxes(title_text=class_name, row=1, col=1)

#     fig.show()

def plot_epoch_log(name, log_df, mode=['loss', 'acc']):
	n_row, n_col = 1, 2
	fig = make_subplots(rows=n_row, cols=n_col, start_cell='top-left', subplot_titles=['Loss', 'Accuracy'])
	use_cols = [[f'{i}_valid', f'{i}_train'] for i in mode]
	
	for i, use_col in enumerate(use_cols):
		for trace in get_traces_by_column(log_df[use_col]):
			fig.add_trace(trace, row=int(i/n_col)+1, col=i%n_col+1)
    
	fig.update_yaxes(title_text=name, row=1, col=1)
	fig.show()

def parse_checkpoint_name_binary(name: str):
	_, epoch, loss = name.split('=')
	epoch = epoch[:2]
	loss = loss[:-5]

	return int(epoch), float(loss)

def load_best_checkpoint_by_class(checkpoint_dir: str):
	cp_name, epoch, loss = '', 0, 1e10

	file_list = os.listdir(checkpoint_dir)
	for f in file_list:
		e, l = parse_checkpoint_name_binary(f)
		if l < loss:
			loss, cp_name, epoch = l, f, e
	
	setting = ExpSetting(log_dir='tmp', batch_size=2)
	
	model = ProtBinaryClass.load_from_checkpoint(setting=setting, checkpoint_path=os.path.join(checkpoint_dir, cp_name))

	return model, epoch, loss

# def load_best_checkpoint_multiclass(checkpoint_dir: str):
