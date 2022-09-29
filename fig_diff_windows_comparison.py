import numpy as np
import matplotlib.pyplot as plt
import os, json

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
        
        # pdb.set_trace()
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        n, bins, _ = ax.hist(durations, bins=list(range(0, int(max(durations)/10)*10+12, 5)), edgecolor="k", density=False, cumulative=False, color="orange")
        ax.set(xlim=(0, 250))
        # plt.xticks(list(range(0, int(max(durations))+20, 20)), fontsize=12)
        plt.xticks([bins[i] for i in range(0, len(bins), 4)], fontsize=15)
        ytick = list(range(0, 130, 20))
        plt.yticks(ytick, fontsize=15)
        ax.set_xlabel(xlabel="Duration (s)", fontsize=20)
        ax.set_ylabel(ylabel="Number of FoG episodes", fontsize=20)
        ax.set_ylim((0, ytick[-1]))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        y = [i/sum(n) for i in n]
        print(x, y, len(x) == len(y))
        ax2 = ax.twinx()
        # ax2.plot(np.array(x), np.array(y)*100, color="b", linewidth=2)
        ax2.set_ylabel("Percent (%)", fontsize=20)
        ax2.grid(axis="y", alpha=1.0)
        # ax2.set_ylim([0, 0.6])
        plt.yticks([yt/len(durations)*100 for yt in ytick], fontsize=15)
        ax2.set_ylim((0, ytick[-1]/len(durations)*100))
        # print([yt/sum(durations) for yt in ytick])
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        # for num, bin in zip(n, bins):
            # ax.annotate(num, xy=(bin, num), xytext=(bin+1.5, num+0.5))
        print("total: ", len(durations), "\nn: \n", n, "\nbins: \n", bins)
        fig.savefig(os.path.join(os.path.dirname(descrip_path), "FoG_hist.png"), dpi=600, bbox_inches="tight")
        plt.close()
        print(int(max(durations)/10)*10+10)

windows = [2, 3, 4, 5, 6, 7, 8]
times = [w*0.5 for w in windows]
sens = [69.26, 74.15, 70.67, 73.35, 74.79, 71.59, 72.98]
spec = [83.56, 86.04, 82.71, 85.83, 87.85, 86.17, 86.83]
M_dr = [95.45, 100.00, 95.45, 100.00, 100.00, 90.00, 100.00]
M_fpr = [45.45, 31.82, 54.55, 22.73, 22.73, 60.00, 50.00]
T_margin = [4.69, 4.25, 6.64, 5.98, 4.91, 6.03, 8.22]

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(np.array(list(range(len(windows)))), np.array(sens), marker="o", color="b", label="Sens", linewidth=3, markersize=15)
ax.plot(np.array(list(range(len(windows)))), np.array(spec), marker="s", color="m", label="Spec", linewidth=3, markersize=15)
ax.plot(np.array(list(range(len(windows)))), np.array(M_dr), marker="p", color="c", label="$M_{dr}$", linewidth=3, markersize=15)
ax.plot(np.array(list(range(len(windows)))), np.array(M_fpr), marker="v", color="r", label="$M_{fpr}$", linewidth=3, markersize=15)

plt.yticks(list(range(0, 110, 20)), fontsize=15)
ax.set_ylabel("Peformance (%)", fontsize=20)
ax.set_ylim((0, 105))
ax.set_xlabel("The number of stacked windows", fontsize=20)
ax.set_xlim((-0.2, len(windows)-0.8))
ax.set_xticks(list(range(len(windows))))
ax.set_xticklabels(windows, fontsize=15)
ax.grid(alpha=1.0)

ax2 = ax.twinx()
ax2.plot(np.array(list(range(len(windows)))), np.array(T_margin), marker="*", color="k", label="$T_{margin}^{d}$", linewidth=3, markersize=15)
ax2.set_ylabel("Time (s)", fontsize=20)
plt.yticks(list(range(0, 11, 2)), fontsize=15)
ax2.set_ylim((1, 10.5))

fig.legend(loc=(0.82, 0.12), fontsize=15, markerscale=0.7, shadow=False, labelspacing=1.5)
fig.savefig("./diff_winds.png", bbox_inches="tight", dpi=300)
plt.close()