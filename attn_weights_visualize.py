import torch, random, os, time, json, pdb
import datetime
import numpy as np
from models import build_model
from dataset import build_dataset, collate_fn, split_n_fold
from utils.plot_utils import plot_logs
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from get_logger import get_root_logger, logger
from test import test
from jsonEncoder import JsonEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# logger = get_root_logger(level=logging.DEBUG, console_out=True, logName="log.log")

class Arguments(object):
    def __init__(self, log_=True) -> None:
        logger.info(f"This machine has {torch.cuda.device_count()} gpu...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rootpath = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets2" ## dataset path
        self.numfolds = 10
        self.seed = 10086
        self.batchsize = 128
        self.epochs = 100
        self.clip_max_norm = 0.15
        self.lr_drop = 30
        self.gamma = 0.5
        self.tag = ""
        self.level = "episode" #"sample"
        self.output_dir = "./outputs_{}_{}".format(self.level, self.tag) if self.tag != "" else "./outputs_{}".format(self.level)
        self.n1 = 3
        self.n2 = 3
        self.resume_fold_split = "./outputs_episode/all_samples.josn"
        if log_:
            self.log()

    def log(self):
        logger.info(f"numfolds: {self.numfolds}, bs: {self.batchsize}, epoch: {self.epochs}, lr_drop: {self.lr_drop}, gamma: {self.gamma}, level: {self.level}, tag{self.tag}, [n1, n2]: [{self.n1}, {self.n2}], seed: {self.seed}")

@torch.no_grad()
def attn_weights_vis():
    args = Arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_chan, d_model, num_class, cls_type, aux_loss = 6, 128, 3, "cls_token", True
    cls_weight, focal_scaler, return_attn_weights = torch.ones(num_class), 2, True
    logger.info(f"in_chan: {in_chan}, d_model: {d_model}, num_class: {num_class}, cls_type: {cls_type}, aux_loss: {aux_loss}")
    # pdb.set_trace()
    samples_split = None
    with open(os.path.join(args.output_dir, "all_samples.josn"), "r") as f:
        samples_split = json.load(f)
    if samples_split is None:
        logger.error("all_sample.json load error.")
        exit()
    ckpt = "checkpoint_0029_00.pth"
    fold = 0
    val_samples = samples_split[args.level]["{}".format(fold)]
    dataset_val = build_dataset(val_samples, rootpth=args.rootpath, level=args.level, mode="val")
    data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)
    model, _ = build_model(in_chan, d_model, num_class, cls_type=cls_type, aux_loss=aux_loss,
                                cls_weight=None, focal_scaler=focal_scaler, return_attn_weights=return_attn_weights)
    model.to(args.device)
    model.eval()
    ckpt = torch.load(os.path.join(args.output_dir, ckpt))
    model.load_state_dict(ckpt["model"])
    pbar = tqdm(data_loader_val)

    predictions, targets, attn_weights = [], [], []
    for samples, t in pbar:
        samples = samples.transpose(1, 2).to(args.device)
        targets.append(t["labels"])

        outputs, attn_weight_now = model(samples)
        predictions.append(outputs["pred_logits"].cpu())
        attn_weights.append(attn_weight_now.cpu())
    # pdb.set_trace()
    targets = torch.cat(targets, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    preds = predictions.argmax(axis=1)
    attn_weights2 = torch.cat(attn_weights, dim=1).numpy() # (num_layers, N, num_heads,L,S)
    # attn_weights = attn_weights2[:, (preds==targets) * (targets == 0)].mean(axis=1)
    # fig_heatmap_with_head_1class(attn_weights, tag="NW")
    # attn_weights = attn_weights2[:, (preds==targets) * (targets == 1)].mean(axis=1)
    # fig_heatmap_with_head_1class(attn_weights, tag="Pre-FoG")
    # attn_weights = attn_weights2[:, (preds==targets) * (targets == 2)].mean(axis=1)
    # fig_heatmap_with_head_1class(attn_weights, tag="FoG")

    fig_heatmap_with_head_3class(attn_weights2[:, (targets == 0)].mean(axis=1),
                                attn_weights2[:, (targets == 1)].mean(axis=1),
                                attn_weights2[:, (targets == 2)].mean(axis=1))

def fig_heatmap_with_head_3class(attn_weights_NW, attn_weights_Pre_FoG, attn_weights_FoG):
    num_heads = attn_weights_NW.shape[1]
    fig, ax = plt.subplots(3, num_heads, figsize=(16, 5))
    for col in range(num_heads): # NW
        sns.heatmap(attn_weights_NW[0, col], ax=ax[0][col], cbar=True, square=False,
                    xticklabels=False, yticklabels=False, cmap="YlGnBu",
                    cbar_kws={"orientation": "vertical", "shrink": 1, "ticks": []})
        if col == 0:
            ax[0][col].set_ylabel("NW", fontsize=17, rotation=90)
    
    for col in range(num_heads): # Pre-FoG
        sns.heatmap(attn_weights_Pre_FoG[0, col], ax=ax[1][col], cbar=True, square=False,
                    xticklabels=False, yticklabels=False, cmap="YlGnBu",
                    cbar_kws={"orientation": "vertical", "shrink": 1, "ticks": []})
        if col == 0:
            ax[1][col].set_ylabel("Pre-FoG", fontsize=17, rotation=90)

    for col in range(num_heads): # FoG
        sns.heatmap(attn_weights_FoG[0, col], ax=ax[2][col], cbar=True, square=False,
                    xticklabels=False, yticklabels=False, cmap="YlGnBu",
                    cbar_kws={"orientation": "vertical", "shrink": 1, "ticks": []})
        if col == 0:
            ax[2][col].set_ylabel("FoG", fontsize=17, rotation=90)
        ax[2][col].set_xlabel("{}".format(col+1), fontsize=14, rotation=0)

    plt.subplots_adjust(wspace=0.03, hspace=0.04)
    fig.supxlabel("Heads of Attention.", fontsize=17)
    fig.savefig("./heatmap/{}.png".format("layer0"), bbox_inches="tight", dpi=600)
    plt.close()

def fig_heatmap_with_head_1class(attn_weights, tag="NW"):
    assert len(attn_weights.shape) == 4
    num_layers, num_heads = attn_weights.shape[:2]
    fig, ax = plt.subplots(1, num_heads, figsize=(16, 3))
    for col in range(num_heads):
        # if col == num_heads-1:
        #     sns.heatmap(attn_weights[0, col], ax=ax[col], cbar=True, cbar_kws={"orientation": "vertical", "shrink": 0.5, "ticks": []},
        #             square=True, cmap="YlGnBu", 
        #             xticklabels=False, yticklabels=False)
        # else:
        sns.heatmap(attn_weights[0, col], ax=ax[col], cbar=True, square=True, cmap="YlGnBu", 
                    xticklabels=False, yticklabels=False, 
                    cbar_kws={"orientation": "vertical", "shrink": 0.38, "ticks": []})
        if col == 0:
            ax[col].set_ylabel(tag, fontsize=10, rotation=90)
        # cbar = ax[col].collections[0].colorbar
        # cbar.ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=None)
    fig.savefig("./heatmap/{}.png".format(tag), bbox_inches="tight", dpi=600)
    plt.close()

def fig_heatmap_with_head2(attn_weights):
    assert len(attn_weights.shape) == 4
    attn_weights = torch.Tensor(attn_weights).softmax(dim=-1).numpy()
    num_layers, num_heads = attn_weights.shape[:2]
    # fig, ax = plt.subplots(num_layers, num_heads, figsize=(18, 12))
    # cbar_ax = plt.axes([0.9, 0.2, 0.05, 0.6])
    for row in range(num_layers):
        for col in range(num_heads):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.heatmap(attn_weights[row, col], ax=ax, cbar=True, square=True, cmap="YlGnBu")    
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            cbar = ax.collections[0].colorbar
            cbar.ax.set_yticklabels([])
            cbar.ax.spines["right"].set_visible(False)
            fig.savefig("./heatmap/heatmap_with_head_{}_{}.png".format(row, col), bbox_inches="tight", dpi=600)
            plt.close()

def fig_heatmap_with_head(attn_weights):
    assert len(attn_weights.shape) == 4
    attn_weights = torch.Tensor(attn_weights).softmax(dim=-1).numpy()
    num_layers, num_heads = attn_weights.shape[:2]

    fig, ax = plt.subplots(num_layers, num_heads, figsize=(18, 12))
    # cbar_ax = plt.axes([0.9, 0.2, 0.05, 0.6])
    for row in range(num_layers):
        for col in range(num_heads):
            if row == 0 and col == 0:
                sns.heatmap(attn_weights[row, col], ax=ax[row][col], cbar=True, square=True, cmap="YlGnBu")    
            else:
                sns.heatmap(attn_weights[row, col], ax=ax[row][col], cbar=True, square=True, cmap="YlGnBu")
            ax[row][col].set_xticklabels([])
            ax[row][col].set_yticklabels([])
            # cbar = ax[row][col].collections[0].colorbar
            # cbar.ax.set_yticklabels([])
    fig.savefig("./heatmap_with_head.png", bbox_inches="tight", dpi=600)
    plt.close()

if __name__ == "__main__":
    # main()
    # args = Arguments()
    # plot_logs(args.output_dir, log_name="train.txt", fields=("loss", "loss_ce", "loss_NW", "loss_preFoG", "loss_FoG", "class_error"))
    attn_weights_vis()

    # import glob
    # args = Arguments(log_=False)
    # # sam_sel = glob.glob(os.path.join(args.output_dir, "011_003_000_*"))
    # sam_sel = [p for p in os.listdir(args.output_dir) if p.startswith("011_003_")]
    # print(sam_sel)