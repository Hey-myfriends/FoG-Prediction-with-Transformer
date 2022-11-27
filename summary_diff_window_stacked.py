import torch
import numpy as np
import os, json
import pdb, glob
import pandas as pd
stepsize = 0.5

def compare_two_model(dict1: dict, dict2: dict): # compare sample level
    cmat1, cmat2 = dict1["cmat"], dict2["cmat"]
    cmat1, cmat2 = np.array(cmat1), np.array(cmat2)
    geom1, geom2 = np.array(dict1["geom"]), np.array(dict2["geom"])
    coef1, coef2 = cmat1.sum(axis=1) / cmat1.sum(), cmat2.sum(axis=1) / cmat2.sum()
    # print(coef1, coef2)
    res1, res2 = (coef1 * geom1).sum(), (coef2 * geom2).sum()
    # print(res1, res2)
    return res1 > res2

def compare_two_model2(dict1: dict, dict2: dict): # compare episode level
    return dict1["false_alarmed"][1] < dict2["false_alarmed"][1] if dict1["false_alarmed"][1] != 0 and dict2["false_alarmed"][1] != 0 else False

def output(metrics_allfold: list):
    accu = sum([m["sampleLevel_best"][0]["accu"] for m in metrics_allfold]) / len(metrics_allfold)
    sens, specs = [], []
    precs, f1_scores = [], []
    detectRate, FalseAlarmRate, PredictedMargin = [], [], []
    for m in metrics_allfold:
        print(m["sampleLevel_best"][1])
        cmat = m["sampleLevel_best"][0]["cmat"]
        cmat = np.array(cmat)
        coef = cmat.sum(axis=1) / cmat.sum()
        sen = np.array(m["sampleLevel_best"][0]["sens"])
        spec = np.array(m["sampleLevel_best"][0]["spec"])
        prec = np.array(m["sampleLevel_best"][0]["prec"])
        f1_score = np.array(m["sampleLevel_best"][0]["f1_score"])
        sens.append((coef * sen).sum())
        specs.append((coef * spec).sum())
        precs.append((coef * prec).sum())
        f1_scores.append((coef * f1_score).sum())

        # pdb.set_trace()
        detectRate.append(m["episodeLevel_best"][0]["detected_FoG"][1])
        FalseAlarmRate.append(m["episodeLevel_best"][0]["false_alarmed"][1])
        # detected_episodes = [v[0] for v in m["episodeLevel_best"][0]["forcast_windows"].values() if isinstance(v[2], int)]
        pred_margin = [v[2] for v in m["episodeLevel_best"][0]["forcast_windows"].values() if isinstance(v[2], int)]
        PredictedMargin.append(sum(pred_margin) * stepsize / len(pred_margin))
    # pdb.set_trace()
    print("sens: ", sens, "\nspecs: ", specs, "\nprecs: ", precs, "\nf1_score: ", f1_score)
    sens, specs = sum(sens) / len(sens), sum(specs) / len(specs)
    precs, f1_scores = sum(precs) / len(precs), sum(f1_scores) / len(f1_scores)
    print("sens_ave: ", sens, "\nspecs_ave: ", specs, "\nprecs_ave: ", precs, "\nf1_scores_ave: ", f1_scores)
    detectRate, FalseAlarmRate, PredictedMargin = sum(detectRate)/len(detectRate), sum(FalseAlarmRate)/len(FalseAlarmRate), sum(PredictedMargin)/len(PredictedMargin)

    res_ave = pd.DataFrame([[accu, sens, specs, detectRate, FalseAlarmRate, PredictedMargin]], 
                    columns=['accu', 'sens', 'specs', 'detectRate', 'FalseAlarmRate', 'PredictedMargin'])
    print(res_ave)

def output2(metrics_allfold: list):
    M_dr, M_pr, M_fpr, M_pr_3, M_pr_1, T_p, T_d = [], [], [], [], [], [], []
    for m in metrics_allfold:
        print(m["episodeLevel_best"][1])
        M_dr.append(m["episodeLevel_best"][0]["detected_FoG"][1])
        M_fpr.append(m["episodeLevel_best"][0]["false_alarmed"][1])

        detected = [v for v in m["episodeLevel_best"][0]["forcast_windows"].values() if isinstance(v[2], int)]
        M_pr.append(len([d[2] for d in detected if d[2] > 0])/m["episodeLevel_best"][0]["total_episodes"])
        M_pr_3.append(len([d[2] for d in detected if d[2] > 3/stepsize])/m["episodeLevel_best"][0]["total_episodes"])
        M_pr_1.append(len([d[2] for d in detected if d[2] > 1/stepsize])/m["episodeLevel_best"][0]["total_episodes"])
        PredMargin = [v[2] for v in m["episodeLevel_best"][0]["forcast_windows"].values() if isinstance(v[2], int) and v[2] > 0]
        T_p.append(sum(PredMargin)/len(PredMargin))
        T_d.append(sum(d[2] for d in detected)/len(detected))
    M_dr, M_pr, M_fpr, M_pr_3, M_pr_1, T_p, T_d = np.array(M_dr)*100, np.array(M_pr)*100, np.array(M_fpr)*100, \
        np.array(M_pr_3)*100, np.array(M_pr_1)*100, np.array(T_p)*stepsize, np.array(T_d)*stepsize
    print(f'\n\nM_dr: {M_dr.mean():.2f}±{M_dr.std():.2f}', f"\nM_pr: {M_pr.mean():.2f}±{M_pr.std():.2f}", f"\nM_fpr: {M_fpr.mean():.2f}±{M_fpr.std():.2f}", 
        f"\nM_pr_3: {M_pr_3.mean():.2f}±{M_pr_3.std():.2f}", f"\nM_pr_1: {M_pr_1.mean():.2f}±{M_pr_1.std():.2f}", 
        f"\nT_p: {T_p.mean():.2f}±{T_p.std():.2f}", f"\nT_d: {T_d.mean():.2f}±{T_d.std():.2f}")

output_dir = "./outputs_episode_with_diff_window_stacked"
jsons = [p for p in os.listdir(output_dir) if p.endswith(".json")]
# print([eval(x.split("_")[-3]) for x in jsons])
jsons = sorted(jsons, key=lambda x: eval(x.split("_")[-3]))

for j in jsons:
    window = j.split("_")[-3]
    with open(os.path.join(output_dir, j), "r") as pf:
        performance = json.load(pf)

    metrics_best = {"sampleLevel_best": None, "episodeLevel_best": None}
    for key, value in performance.items():
        if key.endswith('sampleLevel'):
            if metrics_best["sampleLevel_best"] is None:
                metrics_best["sampleLevel_best"] = [value, key]
            else:
                if compare_two_model(value, metrics_best["sampleLevel_best"][0]):
                    metrics_best["sampleLevel_best"] = [value, key]
        elif key.endswith("episodeLevel"):
            if metrics_best["episodeLevel_best"] is None and value["false_alarmed"][1] != 0:
                metrics_best["episodeLevel_best"] = [value, key]
            else:
                if compare_two_model2(value, metrics_best["episodeLevel_best"][0]):
                    metrics_best["episodeLevel_best"] = [value, key]
        else:
            raise ValueError("metric key error.")

    print("\n\n{} performance: ".format(j))
    output2([metrics_best])
    