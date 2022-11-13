

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
        self.tag = "1_encoderlayer"
        self.level = "episode" #"sample"
        self.output_dir = "./outputs_{}_{}".format(self.level, self.tag)
        self.n1 = 3
        self.n2 = 3
        self.resume_fold_split = "./outputs_episode/all_samples.josn"
        if log_:
            self.log()

    def log(self):
        logger.info(f"numfolds: {self.numfolds}, bs: {self.batchsize}, epoch: {self.epochs}, lr_drop: {self.lr_drop}, gamma: {self.gamma}, level: {self.level}, tag{self.tag}, [n1, n2]: [{self.n1}, {self.n2}], seed: {self.seed}")

def main():
    # pdb.set_trace()
    args = Arguments()
    if not os.path.exists(args.output_dir):
        logger.info("Output dir not exist, make it.")
        os.mkdir(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_chan, d_model, num_class, cls_type, aux_loss = 6, 128, 3, "cls_token", True
    cls_weight, focal_scaler, num_encoder_layers = torch.ones(num_class), 2, 1
    logger.info(f"in_chan: {in_chan}, d_model: {d_model}, num_class: {num_class}, cls_type: {cls_type}, aux_loss: {aux_loss}")

    if args.resume_fold_split is None:
        all_samples = split_n_fold(n = args.numfolds, rootpth=args.rootpath, level=args.level, seed=args.seed)
        samples_dist = {args.level: {fold: all_samples[fold] for fold in range(args.numfolds)}}
        with open(os.path.join(args.output_dir, "all_samples.josn"), "w") as f:
            f.write(json.dumps(samples_dist, ensure_ascii=False, cls=JsonEncoder, indent=4, separators=(",", ":")))
    else:
        with open(args.resume_fold_split, "r") as f:
            all_samples = json.load(f)
            all_samples = [all_samples[args.level][fold] for fold in all_samples[args.level].keys()]

    for fold in range(0, 3): #range(args.numfolds):
        logpath = os.path.join(args.output_dir, f"train_fold_{fold:02}.txt")
        if os.path.exists(logpath):
            os.remove(logpath)
            logger.info("log file already exist, remove it.")
        val_samples = all_samples[fold]
        train_samples = []
        for train_fold in range(args.numfolds):
            if train_fold != fold:
                train_samples.extend(all_samples[train_fold])

        dataset_train = build_dataset(train_samples[:], rootpth=args.rootpath, level=args.level, mode="train")
        dataset_val = build_dataset(val_samples, rootpth=args.rootpath, level=args.level, mode="val")

        data_loader_train = DataLoader(dataset_train, args.batchsize, shuffle=True, collate_fn=collate_fn)
        data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)
        data_loader_val_shuffle = DataLoader(dataset_val, args.batchsize, shuffle=True, collate_fn=collate_fn)

        model, criterion = build_model(in_chan, d_model, num_class, cls_type=cls_type, aux_loss=aux_loss,
                                cls_weight=dataset_train.cls_weight, focal_scaler=focal_scaler, 
                                num_encoder_layers=num_encoder_layers, return_attn_weights=False)
        model.to(args.device)
        criterion.to(args.device)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, 
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], 
            "lr": 1e-3}
            ]
        optimizer = torch.optim.Adam(param_dicts, lr=1e-3, weight_decay=1e-4)
        lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.gamma)
        
        logger.info("Fold {}: Start training...".format(fold))
        start_time = time.time()
        # pdb.set_trace()
        ckpts = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, args.device, 
                epoch, args.clip_max_norm
            )
            lr_schedular.step()

            test_stats = evaluate(
                model, criterion, data_loader_val_shuffle, args.device
            )

            log_stats = {"epoch": epoch,
                        "n_params": n_parameters,
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        }

            if args.output_dir:
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.epochs == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint_{epoch:04}_{fold:02}.pth")
                    torch.save({
                        "fold": fold,
                        "epoch": epoch,
                        "args": args,
                        "model": model.state_dict(),
                        # "stats": log_stats,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_schedular.state_dict()
                    }, ckpt)
                    ckpts.append(f"checkpoint_{epoch:04}_{fold:02}.pth")
                with open(logpath, "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        plot_logs(args.output_dir, log_name=f"train_fold_{fold:02}.txt", fields=("loss", "loss_ce", "loss_NW", "loss_preFoG", "loss_FoG", "class_error"))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Fold.{} Training time cost: {}'.format(fold, total_time_str))

        performance = test(model, data_loader_val, args.output_dir, args.device, level=args.level, ckpts=ckpts, n1=args.n1, n2=args.n2)
        with open(os.path.join(args.output_dir, f"test_fold_{fold:02}.json"), "w") as pf:
            pf.write(json.dumps(performance, ensure_ascii=False, cls=JsonEncoder, indent=4, separators=(",", ":")))

def test_ckpts():
    args = Arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_chan, d_model, num_class, cls_type, aux_loss = 6, 128, 3, "cls_token", True
    cls_weight, focal_scaler = torch.ones(num_class), 2
    logger.info(f"in_chan: {in_chan}, d_model: {d_model}, num_class: {num_class}, cls_type: {cls_type}, aux_loss: {aux_loss}")
    # pdb.set_trace()
    samples_split = None
    with open(os.path.join("./", "all_samples.josn"), "r") as f:
        samples_split = json.load(f)
    if samples_split is None:
        logger.error("all_sample.json load error.")
        exit()
    ckpts = [p for p in os.listdir(args.output_dir) if p.endswith(".pth")]
    # folds = set([int(ckpt.split(".")[0][-2:]) for ckpt in ckpts])
    folds = {}
    for ckpt in ckpts:
        fld = int(ckpt.split(".")[0][-2:])
        if fld not in folds.keys():
            folds[fld] = [ckpt]
        else:
            folds[fld].append(ckpt)
    for fold in folds.keys(): #range(args.numfolds):
        logger.info("Performing fold {} validation.".format(fold))
        val_samples = samples_split[args.level]["{}".format(fold)]
        dataset_val = build_dataset(val_samples, rootpth=args.rootpath, level=args.level, mode="val")
        data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)

        model, _ = build_model(in_chan, d_model, num_class, cls_type=cls_type, aux_loss=aux_loss,
                                cls_weight=None, focal_scaler=focal_scaler)
        model.to(args.device)
        performance = test(model, data_loader_val, args.output_dir, args.device, level=args.level, ckpts=folds[fold], n1=args.n1, n2=args.n2)
        # pdb.set_trace()
        with open(os.path.join(args.output_dir, f"test_alone_fold_{fold:02}.json"), "w") as pf:
            pf.write(json.dumps(performance, ensure_ascii=False, cls=JsonEncoder, indent=4, separators=(",", ":")))

if __name__ == "__main__":
    main()
    # args = Arguments()
    # plot_logs(args.output_dir, log_name="train.txt", fields=("loss", "loss_ce", "loss_NW", "loss_preFoG", "loss_FoG", "class_error"))
    # test_ckpts()

    # import glob
    # args = Arguments(log_=False)
    # # sam_sel = glob.glob(os.path.join(args.output_dir, "011_003_000_*"))
    # sam_sel = [p for p in os.listdir(args.output_dir) if p.startswith("011_003_")]
    # print(sam_sel)