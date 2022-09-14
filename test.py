
import logging
from get_logger import logger, get_root_logger
from torch import nn, Tensor
import torch, os, pdb
from torch.utils.data import DataLoader
from typing import Optional, Iterable
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np

@torch.no_grad()
def test(model: nn.Module, val_dataloader: Iterable, output_dir: str, device: torch.device, level="episode"):
    logger.info("Computing test metrics, level is {}.".format(level))
    ckpts = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith(".pth")]
    performance = {}
    model = model.to(device)
    model.eval()
    # pdb.set_trace()
    for c in ckpts:
        ckpt = torch.load(os.path.join(output_dir, c))
        model.load_state_dict(ckpt["model"])
        pbar = tqdm(val_dataloader)

        predictions, targets, targets_info = [], [], []
        for samples, t in pbar:
            samples = samples.transpose(1, 2).to(device)
            # targets = targets.to(device)
            targets_info.extend(t["info"])
            targets.append(t["labels"])

            outputs = model(samples)
            predictions.append(outputs["pred_logits"].cpu())
        # pdb.set_trace()
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        preds = predictions.argmax(axis=1)

        accu = metrics.accuracy_score(targets, preds)
        f1_score = metrics.f1_score(targets, preds, average=None)
        sens = metrics.recall_score(targets, preds, average=None)
        prec = metrics.precision_score(targets, preds, average=None)
        # auc = metrics.roc_auc_score(targets, preds, average=None, multi_class="ovr")
        auc = None
        geom = np.sqrt(sens * prec)
        cmat = metrics.confusion_matrix(targets, preds)
        samples_level = {"accu": accu, "sens": sens.tolist(), "prec": prec.tolist(), "geom": geom.tolist(), 
                        "auc": auc, "f1_score": f1_score.tolist(), "cmat": cmat.tolist()}
        logger.info("Fold {}, epoch {} sample level performance: ".format(ckpt["fold"], ckpt["epoch"]) + str(samples_level))
        performance.update({"Fold_{}_epoch_{}_sampleLevel".format(ckpt["fold"], ckpt["epoch"]): samples_level})
        # pdb.set_trace()
        if level == "episode":
            episode_metrics_dict = cal_episode_metrics(targets, targets_info, preds)
            logger.info("Fold {}, epoch {} episode level performance: ".format(ckpt["fold"], ckpt["epoch"]) + str(episode_metrics_dict) + "\n\n")
            performance["Fold_{}_epoch_{}_episodeLevel".format(ckpt["fold"], ckpt["epoch"])] = episode_metrics_dict
    return performance

def cal_episode_metrics(targets: np.array, targets_info: list, preds: np.array):
    # parse targets_info to info
    # patients = [info.split("_")[0] for info in targets_info if info.endswith(".txt")]
    # tasks = [info.split("_")[1] for info in targets_info if info.endswith(".txt")]
    # episodes = [info.split("_")[2] for info in targets_info if info.endswith(".txt")]
    episodes = []
    for info in targets_info:
        if not info.endswith(".txt"):
            continue
        p, t, e = info.split("_")[:3]
        episodes.append("{}_{}_{}".format(p, t, e))
    episodes = np.array(episodes)
    detected_FoG, false_alarmed, total_episodes, forcast_windows = 0, 0, 0, []
    curidx = 0
    while curidx < episodes.shape[0]:
        epi_loc = (episodes == episodes[curidx])
        total_episodes += 1
        targets_now, preds_now = targets[epi_loc], preds[epi_loc]
        isDetected, FalseAlarm, PredictMargin = cal_one_episode(targets_now, preds_now)
        logger.info("episode {}, name = {}: isDetected/{}, FalseAlarm/{}, PredictMargin/{}.".format(
            total_episodes, episodes[curidx], isDetected, FalseAlarm, PredictMargin))
        detected_FoG += isDetected
        false_alarmed += FalseAlarm
        forcast_windows.append(PredictMargin)

        curidx += epi_loc.sum()
    return {"detected_FoG": [detected_FoG/1.0, detected_FoG/total_episodes], "false_alarmed": [false_alarmed/1.0, false_alarmed/total_episodes], "total_episodes": total_episodes/1.0, 
            "forcast_windows": forcast_windows}

def cal_one_episode(gts: np.array, preds: np.array, n1=2, n2=3):
    idx_pre, idx_fog = np.argwhere(gts == 1)[0, 0], np.argwhere(gts == 2)[0, 0]
    assert np.all(gts[:idx_pre] == 0) and np.all(gts[idx_pre:idx_fog] == 1) and np.all(gts[idx_fog:] == 2), \
        "please check whether in episode level..."

    hits = np.argwhere(preds != 0)[:, 0]
    hits_step = np.argwhere(hits[1:] - hits[:-1] != 1)[:, 0]
    if hits.shape[0] == 0:
        logger.warning("Note: predictions are all NW.")
        return False, False, 0
    if hits_step.shape[0] == 0:
        logger.warning("predictions only have one step.")
        return True, False, idx_fog # bug
    hits_step = np.concatenate([hits_step, [hits.shape[0]-1]], axis=0)
    start, base = hits[0], hits[0]
    potential = False
    isDetected, FalseAlarm, PredictMargin = False, False, 0
    for point in hits_step:
        if hits[point] < idx_fog:
            if hits[point] - start + 1 >= n1:
                potential = True
                if point + 1 < hits.shape[0] and hits[point+1] - hits[point] >= n2:
                    FalseAlarm = True
                    potential = False
                    base = hits[point + 1]
            else:
                potential = False
                if point + 1 < hits.shape[0]:
                    base = hits[point + 1]
        else:
            if potential or hits[point] - start + 1 >= n1:
                PredictMargin = idx_fog - base
                isDetected = True
                break
        if point + 1 < hits.shape[0]:
            start = hits[point+1]
    return isDetected, FalseAlarm, PredictMargin

if __name__ == "__main__":
    test()