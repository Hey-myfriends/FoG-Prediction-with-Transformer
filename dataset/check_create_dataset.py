import matplotlib.pyplot as plt
import numpy as np
import os, shutil, pdb
import pandas as pd
import json, sys
rootpth = "/home/bebin.huang/Code/FoG_prediction/FoG_prediction-new"
if rootpth not in sys.path:
    sys.path.insert(0, rootpth)

import logging
from get_logger import get_root_logger
logger = get_root_logger(level=logging.DEBUG, console_out=True, logName="./create_datasets.log")
# fmt = "%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.DEBUG, filename="create_datasets.log", filemode="w",
    # format=fmt)

cols = [
        "time", "date", # 0
        'FP1', 'FP2', 'F3', 'F4', 'C4', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 
        'F8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', # 2~26
        'EMG1', 'EMG2', 'IO', 'EMG3', 'EMG4', # 27~31
        'LShankACCX', 'LShankACCY', 'LShankACCZ', 'LShankGYROX', 'LShankGYROY', 'LShankGYROZ', 'NC', 
        'RShankACCX', 'RShankACCY', 'RShankACCZ', 'RShankGYROX', 'RShankGYROY', 'RShankGYROZ', 'NC.1', 
        'WaistACCX', 'WaistACCY', 'WaistACCZ', 'WaistGYROX', 'WaistGYROY', 'WaistGYROZ', 'NC.2', 
        'ArmACCX', 'ArmACCY', 'ArmACCZ', 'ArmGYROX', 'ArmGYROY', 'ArmGYROZ', 'SC', # 32~59
        "Label" # 60
        ]
fs, fs_eeg = 500, 1000 #Hz
window, step = 3, 0.5
pre_FoG = 5
select_cols = ['LShankACCX', 'LShankACCY', 'LShankACCZ', 'LShankGYROX', 'LShankGYROY', 'LShankGYROZ']


def create_datasets():
    rootpath = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets/Filtered Data"
    output_path = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets2"
    if os.path.exists(output_path):
        logger.info("output path exists, delete previous dataset...")
        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
    exclude = ["002", "004", "005", "012"]
    # pdb.set_trace()
    patients = os.listdir(rootpath)
    patients = [p for p in patients if p not in exclude and os.path.isdir(os.path.join(rootpath, p))]

    for patient in patients[:]:
        logger.info("create {} data...".format(patient))
        create_one_patient_data(rootpath, patient, output_path)
        logger.info("create {} data successfully...".format(patient))
    logger.info("Create FoG dataset successfully, fs: {}Hz, window: {}s, step: {}s, pre_FoG: {}s, cols: {}".format(fs, window, step, pre_FoG, select_cols))

def create_one_patient_data(rootpath: str, patientID: str, output_path: str, ignore=False):
    patient_path = os.path.join(rootpath, patientID)
    tasks = os.listdir(patient_path)
    tasks = [f for f in tasks if f.endswith("txt") or os.path.isdir(os.path.join(patient_path, f))]
    windowlength, stepsize, prefog = int(window * fs), int(step * fs), int(pre_FoG * fs)

    if ignore:
        ID_ = rootpath.split("/")[-1]
        # ID_[0] = patientID.split("_")[-1]
        patientID = str(int(ID_)*10+int(patientID[-1])).zfill(3)
    for t, task in enumerate(tasks):
        if os.path.isdir(os.path.join(patient_path, task)):
            create_one_patient_data(patient_path, task, output_path, ignore=True)
            continue
        data = pd.read_csv(os.path.join(patient_path, task), header=None)
        data.columns = cols
        data_select = data.loc[:, select_cols].to_numpy()
        labels = data["Label"].to_numpy()
        z_score(data_select)
        # logger.info(data_select.mean(axis=0), data_select.std(axis=0))
        episodes_one_task = split_to_episode(data_select, labels)
        if episodes_one_task is None:
            logger.warning(f"patient {patientID}/ task {t}: This task does not include qualified FoG episode...")
            continue
        for epi, (d, lab) in enumerate(episodes_one_task):
            onset = np.argwhere(lab == 1)[0, 0]
            if onset < prefog:
                logger.warning("patient {}/ task {}/ episode {}: preFoG shorter than {} sec...".format(patientID, t, epi, pre_FoG))
                continue
            sam_end, sam_num = windowlength, 0
            while sam_end < lab.shape[0]:
                sample_now = d[sam_end-windowlength:sam_end]
                lab_now = determine_label(sam_end, onset, prefog)
                name = "{}_{}_{}_{}_{}.txt".format(patientID, str(t).zfill(3), str(epi).zfill(3), str(sam_num).zfill(3), lab_now) ## save format: patient_task_episode_sample_label.txt
                sam_num += 1
                sam_end += stepsize
                np.savetxt(os.path.join(output_path, name), sample_now)

def determine_label(sam_end: int, onset: int, prefog: int):
    if sam_end < onset - prefog:
        return 0
    elif sam_end > onset - prefog and sam_end < onset:
        return 1
    else:
        if sam_end - onset < int(window * fs / 2):
            return 1
        else:
            return 2

def split_to_episode(data: np.array, labels: np.array):
    if np.all(labels == 0) or np.all(labels == 1):
        return None
    lab_step = labels[1:] - labels[:-1]
    stop = np.argwhere(lab_step == -1)[:, 0]+1
    stop = np.concatenate(([0], stop), axis=0)
    episodes = []
    for i in range(1, stop.shape[0]):
        episodes.append([data[stop[i-1]:stop[i]], labels[stop[i-1]:stop[i]]])
    return episodes

def z_score(data: np.array): # [N, C], in-place
    mu = data.mean(axis=0)
    sig = data.std(axis=0)
    data -= mu[None, ...]
    data /= sig[None, ...]

def check_datasets(save_desrip=False, fig=False):
    rootpath = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets/Filtered Data"
    patients = os.listdir(rootpath)
    patients = [p for p in patients if os.path.isdir(os.path.join(rootpath, p))]
    # print(patients)

    # pdb.set_trace()
    descriptions = {"description": "This file records missing data of each patient."}
    for patient in patients[:]:
        print("check {} data...".format(patient))
        check_one_patient_data(rootpath, patient, descriptions, fig=fig)
    if save_desrip:
        with open(os.path.join(rootpath, "description.json"), "w") as f:
            f.write(json.dumps(descriptions, indent=4, separators=(",", ":")))

def check_one_patient_data(rootpath: str, patientID: str, descriptions: dict, ignore=False, fig=False):
    patient_path = os.path.join(rootpath, patientID)
    files = os.listdir(patient_path)
    files = [f for f in files if f.endswith("txt") or os.path.isdir(os.path.join(patient_path, f))]

    eachtxt = {}
    for file in files:
        if os.path.isdir(os.path.join(patient_path, file)):
            check_one_patient_data(patient_path, file, descriptions, ignore=True, fig=fig)
            continue
        data = pd.read_csv(os.path.join(patient_path, file), header=None)
        data.columns = cols
        data.pop("date")
        zero_col = data.loc[:, (data == 0).all(axis=0)].columns.tolist()
        eachtxt[file] = {"missing data": zero_col}

        details = FoG_details(data)
        eachtxt[file].update(details)
        if fig:
            plot_acc_gyro(data, patient_path, file)
    if ignore:
        name = rootpath.strip().split("/")[-1]+"_"+patientID
        descriptions[name] = eachtxt
    else:
        descriptions[patientID] = eachtxt

def plot_acc_gyro(data: pd.DataFrame, patient_path: str, file: str, 
    fields=('LShankACCX', 'LShankACCY', 'LShankACCZ', 'LShankGYROX', 'LShankGYROY', 'LShankGYROZ', "Label")):

    assert len(fields) > 0, "fields must not be None..."
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 20))
    time = np.arange(data.shape[0]) / fs
    for i in range(len(fields)):
        mu, sig = data[fields[i]].mean(), data[fields[i]].std()
        axes[i].plot(time, data[fields[i]], label="{}_mu={:.2f}_std={:.2f}".format(fields[i], mu, sig))
        axes[i].set(xlim=(time[0], time[-1]))
        axes[i].legend()
    fig.savefig(os.path.join(patient_path, f"{file}.png"), dpi=600, bbox_inches="tight")
    plt.close()

def statistic(): # 选用 LShank ACC+GYRO最合适，需排除 002，004, 005(no FoG)和012三位患者数据
    # pdb.set_trace()
    descrip_path = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets/Filtered Data/description.json"
    with open(descrip_path, "r") as f:
        descriptions = json.load(f)
        # print(descriptions)
        cnt, durations = {}, []
        for key, v in descriptions.items():
            if not isinstance(v, dict) or len(v) == 0:
                continue
            for v1 in v["task_1.txt"]["missing data"]:
                if v1 not in cnt.keys():
                    cnt[v1] = 1
                else:
                    cnt[v1] += 1
            for v2 in v.values():
                durations.extend(v2["durations"])
        cnt = [[k, v] for k, v in cnt.items()]
        cnt = sorted(cnt, key=lambda x: x[1], reverse=True)
        print("missing data deatils: \n\t", cnt)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        n, bins, _ = ax.hist(durations, bins=30, edgecolor="k", density=True, cumulative=False)
        print("total: ", len(durations), "\nn: \n", n, "\nbins: \n", bins)
        fig.savefig(os.path.join(os.path.dirname(descrip_path), "FoG_hist.png"), dpi=600, bbox_inches="tight")
        plt.close()

def FoG_details(data: pd.DataFrame):
    labels = data["Label"].to_numpy()
    lab_step = labels[1:] - labels[:-1]
    start, stop = np.argwhere(lab_step == 1)[:, 0] + 1, np.argwhere(lab_step == -1)[:, 0]
    assert len(start) == len(stop), "the number of start not equal to stop, {} / {} respectively...".format(len(start), len(stop))
    total_episodes = len(start)
    durations = ((stop - start) / fs).tolist()
    details = {"total_FoG_episodes": total_episodes, "durations": durations}
    return details

if __name__ == "__main__":
    create_datasets()
    # check_datasets(fig=False, save_desrip=True)
    # statistic()
    # rootpath = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets/Filtered Data"
    # pdb.set_trace()

    # # data = np.loadtxt(os.path.join(rootpath, "001/task_1.txt"))
    # data = pd.read_csv(os.path.join(rootpath, "001/task_1.txt"), header=None)
    # data.columns = cols
    # data.pop("date")
    # print(data.shape)