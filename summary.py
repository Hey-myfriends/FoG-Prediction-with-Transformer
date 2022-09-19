# import cmath
import torch
import numpy as np
import os, json
import pdb, glob
import pandas as pd


def compare_two_model(dict1: dict, dict2: dict): # compare sample level
    cmat1, cmat2 = dict1["cmat"], dict2["cmat"]
    cmat1, cmat2 = np.array(cmat1), np.array(cmat2)
    geom1, geom2 = np.array(dict1["geom"]), np.array(dict2["geom"])
    coef1, coef2 = cmat1.sum(axis=1) / cmat1.sum(), cmat2.sum(axis=1) / cmat2.sum()
    print(coef1, coef2)
    res1, res2 = (coef1 * geom1).sum(), (coef2 * geom2).sum()
    print(res1, res2)
    return res1 > res2

def compare_two_model2(dict1: dict, dict2: dict): # compare episode level
    return dict1["false_alarmed"][1] < dict2["false_alarmed"][1]

def output(metrics_allfold: list):
    accu = sum([m["sampleLevel_best"][0]["accu"] for m in metrics_allfold]) / len(metrics_allfold)
    sens, specs = [], []
    detectRate, FalseAlarmRate, PredictedMargin = [], [], []
    for m in metrics_allfold:
        cmat = m["sampleLevel_best"][0]["cmat"]
        cmat = np.array(cmat)
        coef = cmat.sum(axis=1) / cmat.sum()
        sen = np.array(m["sampleLevel_best"][0]["sens"])
        spec = np.array(m["sampleLevel_best"][0]["spec"])
        sens.append((coef * sen).sum())
        specs.append((coef * spec).sum())

        # pdb.set_trace()
        detectRate.append(m["episodeLevel_best"][0]["detected_FoG"][1])
        FalseAlarmRate.append(m["episodeLevel_best"][0]["false_alarmed"][1])
        pred_margin = [v[2] for v in m["episodeLevel_best"][0]["forcast_windows"].values() if isinstance(v[2], int)]
        PredictedMargin.append(sum(pred_margin)/len(pred_margin))
    # pdb.set_trace()
    sens, specs = sum(sens) / len(sens), sum(specs) / len(specs)
    detectRate, FalseAlarmRate, PredictedMargin = sum(detectRate)/len(detectRate), sum(FalseAlarmRate)/len(FalseAlarmRate), sum(PredictedMargin)/len(PredictedMargin)

    res = pd.DataFrame([[accu, sens, specs, detectRate, FalseAlarmRate, PredictedMargin]], 
                    columns=['accu', 'sens', 'specs', 'detectRate', 'FalseAlarmRate', 'PredictedMargin'])
    print(res)

outputs_path = "./outputs_episode_with_dataAug"
json_prefix = "test_alone*.json"

# json_all = [p for p in os.listdir(outputs_path) if p.startswith(json_prefix)]
json_all = glob.glob(os.path.join(outputs_path, json_prefix))
# pdb.set_trace()
metrics_allfolds = []
for fold_path in json_all:
    with open(fold_path, "r") as f:
        metrics_fold = json.load(f)

    metrics_best = {"sampleLevel_best": None, "episodeLevel_best": None}
    for key, value in metrics_fold.items():
        if key.endswith('sampleLevel'):
            if metrics_best["sampleLevel_best"] is None:
                metrics_best["sampleLevel_best"] = [value, key]
            else:
                if compare_two_model(value, metrics_best["sampleLevel_best"][0]):
                    metrics_best["sampleLevel_best"] = [value, key]
        elif key.endswith("episodeLevel"):
            if metrics_best["episodeLevel_best"] is None:
                metrics_best["episodeLevel_best"] = [value, key]
            else:
                if compare_two_model2(value, metrics_best["episodeLevel_best"][0]):
                    metrics_best["episodeLevel_best"] = [value, key]
        else:
            raise ValueError("metric key error.")

    metrics_allfolds.append(metrics_best)
output(metrics_allfolds)

