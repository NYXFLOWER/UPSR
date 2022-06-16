#%%
import os
os.chdir('/home/flower/github/Protein3D/Protein3D')

from utils import *
from torch.cuda import max_memory_allocated
from torch.nn.functional import softmax

from models import *
# from pytorch_memlab import LineProfiler, MemReporter, profile, profile_every, set_target_gpu

#%%


NUM_FUNC = 384


# @profile
# def main():
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

setting = ExpSetting(
    data_address='../data/ProtFunct_dict.pt',
    distance_cutoff=[3, 3.5], 
    num_layers=2,
    atom_feature_size=20,
    num_degrees=3, 
    seed=1111,
    num_channels=4, 
    head=1, 
    # use_classes = [357, 351, 23, 215, 190, 162, 62, 195, 321, 369],
    # use_classes = list(range(NUM_FUNC)),
    use_classes = [357, 351, 0],
    # use_classes = [162, 62, 195, 321, 369],
    batch_size=32,
    # decoder_mid_dim=64,
    lr=1e-3
    )
max_epoch = 500

seed_all(setting.seed)

model = ProtMultClassModel(setting)
model.to(device)

train_loader, valid_loader, test_loader = load_data_ie(setting)

optimizer = torch.optim.Adam(model.parameters(), setting.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, setting.num_epochs, eta_min=1e-4)

# c_recaster = get_class_recaster(setting.use_classes)
loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = tm.Accuracy(num_classes=setting.num_class).to(device)
auroc_fn = tm.AUROC(num_classes=setting.num_class).to(device)
acc_fn2 = tm.Accuracy(top_k=2).to(device)

# reporter = MemReporter()
# reporter.report(verbose=True)

#%%
# @profile
def main():
    for e in range(max_epoch):

        for i, batch in enumerate(valid_loader):

            # if i < 420:
            #     continue

            g, targets, pdb = batch
            g = g.to(device)
            targets = targets.to(device)

            model.train()

            # if i == 0:
            #     reporter.report(verbose=True)

            optimizer.zero_grad()

            # if i == 0:
            #     with LineProfiler(model(g)) as prof:
            #         prof.display()

            z = model(g)

            # if i == 0:
            #     reporter.report(verbose=True)

            loss = loss_fn(z, targets)

            loss.backward()

            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                print(f"{e:3} - {i:3}, train loss: {loss}")
                break


        if e % 50 == 0:
            val_test(e, i, model)
        
def val_test(e, i, model):
    model.eval()
    p_list = []
    target_list = []
    for loader in [test_loader]:
        p_list = []
        pdb_list = []
        target_list = []
        for batch in loader:
            g, targets, pdb = batch
            g = g.to(device)
            targets = targets.to(device)
            pre_p = model(g)
            p_list.append(pre_p)
            target_list.append(targets)
            pdb_list.append(pdb)
        # print('done')

        pdb_list = np.concatenate(pdb_list)
        pre_ps = torch.cat(p_list)
        preds = torch.argmax(pre_ps, dim=1)
        ts = torch.cat(target_list)

        loss = loss_fn(pre_ps, ts)
        loss_num = loss.detach().tolist()
        acc = acc_fn(preds, ts)
        auroc = auroc_fn(pre_ps, ts)
        acc2 = acc_fn2(pre_ps, ts)

        print(f"{e:3} - {i:3}, loss:{loss:.4}, acc: {acc:.4}, acc_top2: {acc2:.4} auroc: {auroc:.4}")
        

# try:
main()

# except:
#     name = f"batch{setting.batch_size}_nl{setting.num_layers}_nd{setting.num_degrees}_nc{setting.num_channels}_dd{setting.decoder_mid_dim}"
#     torch.save(model, f"../model/{name}.pt") 

#%%





#%%
# name = f"ec{len(setting.use_classes)}_batch{setting.batch_size}_nl{setting.num_layers}_nd{setting.num_degrees}_nc{setting.num_channels}_dd{setting.decoder_mid_dim}"
# torch.save(model, f"../model/{name}.pt")


# #%%
# # -------------- analysis --------------
# import pandas as pd
# ecs = pd.read_csv('../data/ProtFunct/unique_functions.txt', header=None).to_numpy().flatten()
# re = []
# for i in range(len(setting.use_classes)):
#     pi = preds[ts == i]
#     acc = torch.sum(pi==i).type(torch.DoubleTensor) / pi.shape[0]
#     ppi = pre_ps[ts == i]
#     acc2 = acc_fn2(ppi, ts[ts == i])
#     re.append([i, ecs[setting.use_classes[i]], pi.shape[0], acc.item(), acc2.item()])
# print(re)
# df = pd.DataFrame(re, columns=['ec_index', 'ec_class', '#testing sample', 'accuracy', 'acc_top_2'])

# m = nn.Softmax(dim=1)
# a = m(pre_ps).cpu().detach().numpy()
# t = ts.cpu().detach().numpy()
# b = np.concatenate([pdb_list.reshape((-1, 1)), t.reshape((-1, 1)), a], axis=1)

# df_pred_full = pd.DataFrame(b, columns=['PDB', 'EC_idx']+[f'idx-{i}: {j}' for i, j in enumerate(df.ec_class.tolist())])
# df_pred_full.to_csv('top5_indiv_pred.csv')


# # for parameter in model.parameters():
# #     print(parameter)

# model = torch.load(f"../model/{name}.pt")
# %%
