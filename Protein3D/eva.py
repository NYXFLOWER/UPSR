from models import *
import pandas as pd
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.callbacks import ModelCheckpoint
# from multi_run import ProtMultClassLitModel
# from run import val_test

# from pytorch_memlab import LineProfiler, MemReporter, profile

# name = "/home/flower/github/Protein3D/Protein3D/lightning_logs/pl_checkpoint/ecall_batch512_nl2_nd3_nc8_nh1/epoch=36-valid_loss_epoch=5.25-accuracy=0.00.ckpt"

# @profile
def main():
    name = '/home/flower/github/Protein3D/Protein3D/lightning_logs/pl_checkpoint/ecall_batch512_nl2_nd3_nc6_nh1/epoch=49-valid_loss_epoch=5.36-accuracy=0.00.ckpt'
    minfo = name.split('/')[-2].split('_')

    setting = ExpSetting(
        distance_cutoff=[3, 3.5], 
        num_layers=2,
        num_degrees=3, 
        seed=1111,
        num_channels=int(minfo[-2][-1:]), 
        num_workers=0,
        head=1, 
        # use_classes = [i for i in range(50)],
        batch_size=1,
        lr=1e-3,
        num_epochs=50
        )

    seed_all(setting.seed)

    _, _, test_loader = load_data(setting)

    model = ProtMultClassLitModel.load_from_checkpoint(setting=setting, checkpoint_path=name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = tm.Accuracy(num_classes=setting.num_class).to(device)
    # auroc_fn = tm.AUROC(num_classes=setting.num_class).to(device)
    # acc_fn2 = tm.Accuracy(top_k=2).to(device)

    # def val_test(e, i, model):

    def write2file(cont):
        with open('eva_out.csv', 'a') as f:
            f.write(f'{cont}\n')


    model.eval()
    p_list = []
    target_list = []
    # for loader in test_loader:
        # p_list = []
        # # pdb_list = []
        # target_list = []
    for batch in test_loader:
        g, targets, pdb = batch
        g = g.to(device)
        targets = targets.to(device)
        pre_p = model(g)
        # p_list.append(pre_p)
        # target_list.append(targets)
        # pdb_list.append(pdb)

        m = nn.Softmax(dim=1)
        a = m(pre_p).cpu().detach().numpy()
        t = targets.cpu().detach().numpy()
        b = np.concatenate([np.array(pdb).reshape((-1, 1)), t.reshape((-1, 1)), a], axis=1)
        df = pd.DataFrame(b)
        df.to_csv('my_csv.csv', mode='a', header=False, index=False)

        

    rdata = pd.read_csv('my_csv.csv', header=None)
    pred_ps = torch.tensor(rdata.values[:, 2:].astype(np.float32))
    preds = torch.argmax(pred_ps, dim=1)
    targets = torch.tensor(rdata[1].values) 

    acc_fn = tm.Accuracy(num_classes=setting.num_class)
    auroc_fn = tm.AUROC(num_classes=setting.num_class)
    acc_fn2 = tm.Accuracy(top_k=2)

    acc = acc_fn(preds, torch.tensor(rdata[1].values))
    auroc = auroc_fn(pred_ps, targets)
    acc2 = acc_fn2(pred_ps, targets)

    print(acc.item(), acc2.item(), auroc.item())

    accl = []
    for i in range(setting.num_class):
        pi = preds[targets == i]
        ac = torch.sum(pi == i).type(torch.DoubleTensor)/pi.shape[0]
        accl.append(ac.item())

    print(np.array(accl).mean())


    print('done')


    

    # # pdb_list = np.concatenate(pdb_list)
    # pre_ps = torch.cat(p_list)
    # preds = torch.argmax(pre_ps, dim=1)
    # ts = torch.cat(target_list)

    # loss = loss_fn(pre_ps, ts)
    # # loss_num = loss.detach().tolist()
    # acc = acc_fn(preds, ts)
    # # auroc = auroc_fn(pre_ps, ts)
    # # acc2 = acc_fn2(pre_ps, ts)

    # print(f"loss:{loss:.4}, acc: {acc:.4}")
        # , acc_top2: {acc2:.4} auroc: {auroc:.4}")

    # val_test(0, 0, model)

main()

