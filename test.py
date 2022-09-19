
import matplotlib.pyplot as plt
from get_logger import logger, get_root_logger
from torch import nn, Tensor
import torch, os, pdb
from torch.utils.data import DataLoader
from typing import Optional, Iterable
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np

@torch.no_grad()
def test(model: nn.Module, val_dataloader: Iterable, output_dir: str, device: torch.device, level="episode", ckpts=None,
        n1=3, n2=3):
    logger.info("Computing test metrics, level is {}.".format(level))
    if ckpts is None:
        ckpts = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith(".pth")]
    ckpts = sorted(ckpts, key=lambda x: x.split(".")[0][-7:])
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
        cmat = metrics.confusion_matrix(targets, preds)
        spec = cal_spec(cmat)
        geom = np.sqrt(sens * spec)
        samples_level = {"accu": accu, "sens": sens.tolist(), "spec": spec.tolist(), "prec": prec.tolist(), "geom": geom.tolist(), 
                        "auc": auc, "f1_score": f1_score.tolist(), "cmat": cmat.tolist()}
        logger.info("Fold {}, epoch {} sample level performance: ".format(ckpt["fold"], ckpt["epoch"]) + str(samples_level))
        performance.update({"Fold_{}_epoch_{}_sampleLevel".format(ckpt["fold"], ckpt["epoch"]): samples_level})
        # pdb.set_trace()
        if level == "episode":
            episode_metrics_dict = cal_episode_metrics(targets, targets_info, preds, output_dir, n1=n1, n2=n2, header="Fold_{}_epoch_{}".format(ckpt["fold"], ckpt["epoch"]))
            logger.info("Fold {}, epoch {} episode level performance: ".format(ckpt["fold"], ckpt["epoch"]) + str(episode_metrics_dict) + "\n\n")
            performance["Fold_{}_epoch_{}_episodeLevel".format(ckpt["fold"], ckpt["epoch"])] = episode_metrics_dict
    return performance

def cal_spec(cmat: np.array):
    specs = []
    ALL = np.sum(cmat)
    for i in range(cmat.shape[0]):
        TP_i = cmat[i, i]
        FN_i = np.sum(cmat[i, :]) - TP_i
        FP_i = np.sum(cmat[:, i]) - TP_i
        TN_i = ALL - TP_i - FP_i - FN_i
        specs.append(TN_i/(TN_i+FP_i))
    return np.array(specs)

def cal_episode_metrics(targets: np.array, targets_info: list, preds: np.array, output_dir: str, n1=3, n2=3, header=""):
    # parse targets_info to info
    episodes = []
    for info in targets_info:
        if not info.endswith(".txt"):
            continue
        p, t, e = info.split("_")[:3]
        episodes.append("{}_{}_{}".format(p, t, e))
    episodes = np.array(episodes)
    detected_FoG, false_alarmed, total_episodes, forcast_windows = 0, 0, 0, {}
    curidx = 0
    while curidx < episodes.shape[0]:
        epi_loc = (episodes == episodes[curidx])
        total_episodes += 1
        targets_now, preds_now = targets[epi_loc], preds[epi_loc]
        
        logger.info("Current episode is {}/{}".format(total_episodes, episodes[curidx]))
        isDetected, FalseAlarm, PredictMargin, figinfo = cal_one_episode(targets_now, preds_now, n1=n1, n2=n2, header="episode {}, name = {}: ".format(total_episodes, episodes[curidx]))
        fig_one_episode(targets_now, preds_now, episodes[curidx], output_dir, header=header, figinfo=figinfo,
                        text="isDetected/{}, FalseAlarm/{}, PredictMargin/{}".format(isDetected, FalseAlarm, PredictMargin))
        if PredictMargin == "no pre-FoG":
            total_episodes -= 1
            curidx += epi_loc.sum()
            continue
        elif PredictMargin == "no FoG":
            total_episodes -= 1
            curidx += epi_loc.sum()
            continue
            
        logger.info("episode {}, name = {}: isDetected/{}, FalseAlarm/{}, PredictMargin/{}, [idx_fog, base]: {}.".format(
            total_episodes, episodes[curidx], isDetected, FalseAlarm, PredictMargin, str(figinfo)))
        detected_FoG += isDetected
        false_alarmed += FalseAlarm
        forcast_windows[episodes[curidx]] = [isDetected, FalseAlarm, PredictMargin, str(figinfo)]

        curidx += epi_loc.sum()
    return {"detected_FoG": [detected_FoG/1.0, detected_FoG/total_episodes], "false_alarmed": [false_alarmed/1.0, false_alarmed/total_episodes], "total_episodes": total_episodes/1.0, 
            "forcast_windows": forcast_windows}

def fig_one_episode(targets_now: np.array, preds_now: np.array, epi_info: str, output_dir: str, header="", text="", figinfo=None):
    targets = targets_now.copy()
    preds = preds_now.copy()
    save_pth = os.path.join(output_dir, "episode_fig/"+header if len(header) != 0 else "episode_fig")
    os.makedirs(save_pth, exist_ok=True)

    fig, ax = plt.subplots(1,1, figsize=(8, 5))
    x = np.arange(targets.shape[0])
    ax.plot(x, targets, color="r", marker=".", label="gt")
    ax.plot(x, preds+2.5, color="k", marker=".", label="dt")
    if figinfo is not None:
        text += ", \nidx_fog/{}, base/{}.".format(figinfo[0], figinfo[1])
        for pos in figinfo:
            ax.axvline(pos, color="grey", linestyle="--")
    ax.legend()
    # ax.set_title(epi_info)
    ax.set(xlim=(x[0], x[-1]), title=epi_info, ylim=(0, 7))
    ax.text(0.5, 6, text, fontsize=10, color="k")
    fig.savefig(os.path.join(save_pth, epi_info), dpi=300, bbox_inches="tight")
    plt.close()

def cal_one_episode(gts: np.array, preds: np.array, n1=3, n2=3, header=""):
    # pdb.set_trace()
    idx_pre, idx_fog = np.argwhere(gts == 1), np.argwhere(gts == 2)
    if idx_pre.shape[0] == 0:
        logger.warning(header + "This episode does not have pre-FoG stage, gts = {}".format(str(gts)))
        return False, False, "no pre-FoG", None
    if idx_fog.shape[0] == 0:
        logger.warning(header + "This episode does not have FoG stage, gts = {}".format(str(gts)))
        return False, False, "no FoG", None
    idx_pre, idx_fog = idx_pre[0, 0], idx_fog[0, 0]
    assert np.all(gts[:idx_pre] == 0) and np.all(gts[idx_pre:idx_fog] == 1) and np.all(gts[idx_fog:] == 2), \
        "please check whether in episode level..."

    hits = np.argwhere(preds != 0)[:, 0]
    hits_step = np.argwhere(hits[1:] - hits[:-1] != 1)[:, 0]
    if hits.shape[0] == 0:
        logger.warning(header + "predictions are all NW.")
        return False, False, "all NW", None
    if hits_step.shape[0] == 0:
        logger.warning(header + "predictions only have one step.")
        return True, False, idx_fog, (idx_fog, hits[0]) # bug
    hits_step = np.concatenate([hits_step, [hits.shape[0]-1]], axis=0)
    start, base = hits[0], hits[0]
    potential = False
    isDetected, FalseAlarm, PredictMargin, figinfo = False, False, 0, None
    for point in hits_step:
        if hits[point] < idx_fog:
            if hits[point] - start + 1 >= n1:
                potential = True
                if point + 1 < hits.shape[0] and hits[point+1] - hits[point] - 1 >= n2:
                    FalseAlarm = True
                    potential = False
                    base = hits[point + 1]
                elif point + 1 >= hits.shape[0]:
                    isDetected, FalseAlarm, PredictMargin = True, False, idx_fog - base
                    figinfo = (idx_fog, base)
                    break
            else:
                if start > base:
                    potential = True
                    if point + 1 < hits.shape[0] and hits[point+1] - hits[point] - 1 >= n2:
                        FalseAlarm = True
                        potential = False
                        base = hits[point + 1]
                    elif point + 1 >= hits.shape[0]:
                        isDetected, FalseAlarm, PredictMargin = True, False, idx_fog - base
                        figinfo = (idx_fog, base)
                        break
                else:
                    potential = False
                    if point + 1 < hits.shape[0]:
                        base = hits[point + 1]
        else:
            if potential or hits[point] - start + 1 >= n1:
                PredictMargin = idx_fog - base
                figinfo = (idx_fog, base)
                isDetected = True
                break
        if point + 1 < hits.shape[0]:
            start = hits[point+1]
        
    return isDetected, FalseAlarm, PredictMargin if isDetected else "not detected", figinfo

if __name__ == "__main__":
    test()