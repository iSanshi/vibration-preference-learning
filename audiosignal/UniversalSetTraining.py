# Case 1 
# Amplitude 2 Frequency 4 Pulsy 4 Envelop 2
import numpy as np
from itertools import product
from itertools import combinations as iter_combinations
from GP_apl import GaussianProcess
import scipy.optimize as opt
from scipy.io import wavfile
import sounddevice as sd
import time
import generateSignal6params
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from sklearn.cluster import KMeans

############################################
# 1. 定义 Tee 类，用来实现双输出（终端 + 文件）
############################################
class Tee(object):
    """
    同时将输出写入日志文件和原始 stdout，让用户在终端还能看到提示。
    """
    def __init__(self, file_path, mode="w", encoding="utf-8"):
        self.file = open(file_path, mode=mode, encoding=encoding)
        self.stdout = sys.stdout  # 原始标准输出

    def write(self, data):
        self.file.write(data)     # 写入日志文件
        self.stdout.write(data)   # 写入终端

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


############################################
# 2. 与实验相关的函数：和之前相同
############################################

def create_tone_wav(filler_amplitude, filler_frequency, filler_density, filler_env_gradient, duration, cycles, fs=44100):
    time_vals, data, _ = generateSignal6params.generate_tone_signal(
        filler_amplitude=filler_amplitude,
        filler_frequency=filler_frequency,
        filler_density=filler_density,
        filler_env_gradient=filler_env_gradient,
        duration=duration,
        cycles=cycles,
        fs=fs
    )
    return time_vals, data

def play_wav(filename, blocking=True):
    try:
        fs, data = wavfile.read(filename)
        normalized_data = data.astype(np.float32) / 32767.0
        sd.play(normalized_data, fs)
        if blocking:
            sd.wait()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def generate_and_play(filler_amplitude, filler_frequency, filler_density, filler_env_gradient, duration, cycles, fs, temp_file):
    time_vals, data = create_tone_wav(
        filler_amplitude,
        filler_frequency,
        filler_density,
        filler_env_gradient,
        duration,
        cycles,
        fs
    )
    normalized_data = np.int16(data * 32767)
    wavfile.write(temp_file, int(fs), normalized_data)
    
    # 播放音频
    play_wav(temp_file)
    
    # 保存波形图，而不显示
    plot_filename = temp_file.replace(os.sep + "wav" + os.sep, os.sep + "png" + os.sep)
    plot_filename = plot_filename.replace(".wav", ".png")
    plt.figure()
    plt.plot(time_vals, data, 'b-')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Sine Wave')
    plt.grid(True)
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def generate_parameter_combinations1():
    amplitude = [30, 60]
    frequency = [20, 40, 60, 80]
    density = [5, 25, 50, 80]
    gradient = [-50, 50]
    combinations = list(product(amplitude, frequency, density, gradient))
    combinations_array = np.array(combinations)
    assert len(combinations_array) == 64, "Expected 64 combinations"
    return combinations_array

def generate_parameter_combinations2():
    amplitude = [30, 60]
    frequency = [25, 50, 75]
    density = [10, 50, 90]
    gradient = [-50, 50]
    combinations = list(product(amplitude, frequency, density, gradient))
    combinations_array = np.array(combinations)
    assert len(combinations_array) == 36, "Expected 36 combinations"
    return combinations_array

def print_combinations(combinations):
    print("Index  [A,    F,    D,    G]")
    print("-" * 30)
    for i, combo in enumerate(combinations):
        print(f"{i:3d}    {combo}")

############################################
# 3. 自定义函数：MaxMin选点 & KMeans
############################################

def maxmin_selection(parameter_combinations, num_points, random_state=None):
    """
    在离散空间里使用 MaxMin 距离最大化选点的简单实现。
    1. 随机选一个起始点
    2. 逐个选取与当前已选集合距离最小值最大的点
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    if num_points > len(parameter_combinations):
        raise ValueError("num_points cannot exceed the total number of available combinations")
    
    # 打乱后先随机挑一个点作为初始点
    shuffled = np.random.permutation(parameter_combinations)
    selected = [shuffled[0]]  # list of arrays
    remaining = list(shuffled[1:])
    
    while len(selected) < num_points and remaining:
        # 计算 remaining 中每个点到已选点集合的最小距离
        max_dist = -1
        next_idx = 0
        for i, cand in enumerate(remaining):
            dists = [np.linalg.norm(cand - s) for s in selected]
            cand_min_dist = min(dists)
            if cand_min_dist > max_dist:
                max_dist = cand_min_dist
                next_idx = i
        # 将距离最大的点加入 selected
        selected.append(remaining[next_idx])
        # 删除该点
        remaining.pop(next_idx)
    
    return np.array(selected)

def pair_points_evenly(selected_points, num_pairs):
    """
    将选好的点做简单配对（打乱后两两分组），得到 num_pairs 对。
    如果点数不够 2 * num_pairs，则会减少到能完整配对的对数。
    """
    np.random.shuffle(selected_points)
    needed = 2 * num_pairs
    if needed > len(selected_points):
        needed = len(selected_points)
    used_points = selected_points[:needed]
    result = np.zeros((needed // 2, 2, used_points.shape[1]))
    idx = 0
    for i in range(0, needed, 2):
        if i+1 < needed:
            result[idx, 0] = used_points[i]
            result[idx, 1] = used_points[i+1]
            idx += 1
    return result

def get_pair_combinations_maxmin(parameter_combinations, num_pairs=20, random_state=None):
    """
    使用 MaxMin 距离最大化选出 2*num_pairs 个点，再将其配对。
    """
    # 第一步，使用 maxmin_selection
    # 我们要 2 * num_pairs 个点
    points_needed = 2 * num_pairs
    selected_points = maxmin_selection(parameter_combinations, points_needed, random_state=random_state)
    # 第二步，配对
    pairs = pair_points_evenly(selected_points, num_pairs)
    return pairs

def get_kmeans_signals(parameter_combinations, num_clusters=16, random_state=42):
    """
    在给定的离散参数组合上运行K-Means，返回 num_clusters 个最具代表性的信号。
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans.fit(parameter_combinations)  # 形状: (N, 4)
    centroids = kmeans.cluster_centers_  # (num_clusters, 4)
    cluster_points = []
    
    # 为每个质心找到在离散空间中距离它最近的点
    for center in centroids:
        distances = np.sum((parameter_combinations - center)**2, axis=1)
        nearest_idx = np.argmin(distances)
        cluster_points.append(parameter_combinations[nearest_idx])
    return np.array(cluster_points)

############################################
# 4. user_study 函数：在Shuffle Comparison阶段用MaxMin，在最终评估阶段用Kmeans
############################################
def user_study(filename):
    """
    1) 将所有print既输出到终端也写进 filename.log
    2) wav & png分别存放
    3) Shuffle Comparison 和 Ranked Comparison 的循环索引从 1 开始
    4) 在所有输入处，打印 "User input: XXX"
    5) 保存 gp.updateParameters 的数据到 gp_updates.csv
    """
    # 准备母文件夹 results
    results_root = "./results"
    os.makedirs(results_root, exist_ok=True)

    # 创建当前实验文件夹
    output_dir = os.path.join(results_root, str(filename))
    os.makedirs(output_dir, exist_ok=True)

    # 分别新建 wav & png 文件夹
    wav_dir = os.path.join(output_dir, "wav")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # 打开日志文件，并使用 Tee 让输出同时写入日志与终端
    log_file_path = os.path.join(output_dir, f"{filename}.log")
    tee = Tee(log_file_path, mode="w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = tee  # 替换 stdout，以便 print 同时输出到文件和终端

    ############################################
    # 以下为原有的逻辑
    ############################################
    gp = GaussianProcess([0,0,0,0], 1.0, 0.1)
    combinations1 = generate_parameter_combinations1()
    np.random.seed(int(time.time()))
    
    # ===================================================
    # 1) Shuffle Comparison 阶段：用 MaxMin 距离最大化选点配对
    # ===================================================
    # 需要 20 组数据 => num_pairs = 20
    iternum = 20
    pair_combinations = get_pair_combinations_maxmin(combinations1, num_pairs=iternum, random_state=None)

    # 初始化数据存储
    user_selections = []
    selected_signal = []
    gp_updates = []  # 新增：用于存储 gp.updateParameters 的数据

    # -------------------------------------
    # Shuffle Comparison 阶段：逐组对比
    # -------------------------------------
    for i in range(iternum):
        round_num = i + 1  # 索引从1开始
        while True:
            print(f"\nMenu（Shuffle Comparison）:{round_num}")
            print("1: Play Signal 1")
            print("2: Play Signal 2")
            print("3: Select Signal 1 and Exit")
            print("4: Select Signal 2 and Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            print("User input:", choice)  # 记录输入
            if choice == "1":
                temp_file = os.path.join(wav_dir, f"shuffle_{round_num}_tone1.wav")
                generate_and_play(
                    filler_amplitude=pair_combinations[i][0][0],
                    filler_frequency=pair_combinations[i][0][1],
                    filler_density=pair_combinations[i][0][2],
                    filler_env_gradient=pair_combinations[i][0][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
                user_selections.append("Played Signal 1:" + str(pair_combinations[i][0]))
            
            elif choice == "2":
                temp_file = os.path.join(wav_dir, f"shuffle_{round_num}_tone2.wav")
                generate_and_play(
                    filler_amplitude=pair_combinations[i][1][0],
                    filler_frequency=pair_combinations[i][1][1],
                    filler_density=pair_combinations[i][1][2],
                    filler_env_gradient=pair_combinations[i][1][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
                user_selections.append("Played Signal 2:" + str(pair_combinations[i][1]))
            
            elif choice == "3":
                selected_signal.append(1)
                user_selections.append("Selected Signal 1：" + str(pair_combinations[i][0]))
                break
                
            elif choice == "4":
                selected_signal.append(-1)
                user_selections.append("Selected Signal 2：" + str(pair_combinations[i][1]))
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # 更新 GP 参数
        gp.updateParameters([pair_combinations[i][0], pair_combinations[i][1]], selected_signal[i])
        # 保存 gp.updateParameters 的数据
        gp_updates.append({
            'Round': round_num,
            'Pair1_A': pair_combinations[i][0][0],
            'Pair1_F': pair_combinations[i][0][1],
            'Pair1_D': pair_combinations[i][0][2],
            'Pair1_G': pair_combinations[i][0][3],
            'Pair2_A': pair_combinations[i][1][0],
            'Pair2_F': pair_combinations[i][1][1],
            'Pair2_D': pair_combinations[i][1][2],
            'Pair2_G': pair_combinations[i][1][3],
            'Selection': selected_signal[i]
        })

    # -------------------------------------
    # Ranked Comparison 阶段
    # -------------------------------------
    for i in range(1, 21):
        max_reward_index = 0
        for j in range(len(combinations1)):
            meancalculation = gp.mean1pt(combinations1[j])
            if meancalculation > gp.mean1pt(combinations1[max_reward_index]):
                max_reward_index = j
        print("max reward:" + str(combinations1[max_reward_index]))
        
        maxinfogain_index = 0
        min_infogain = gp.objectiveEntropy(np.append(combinations1[max_reward_index], combinations1[max_reward_index]))
        print("maxinfogain:" + str(min_infogain))
        
        for k in range(len(combinations1)):
            combined = np.append(combinations1[max_reward_index], combinations1[k])
            infogain = gp.objectiveEntropy(combined)
            if infogain > min_infogain:
                print("found a larger infogain:" + str(infogain))
                min_infogain = infogain
                maxinfogain_index = k
                print(combinations1[k])
        print("max_infogain_index:" + str(maxinfogain_index))
        
        while True:
            print(f"\nMenu(Ranked Comparison):{i}")
            print("1: Play Signal 1")
            print("2: Play Signal 2")
            print("3: Select Signal 1 and Exit")
            print("4: Select Signal 2 and Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            print("User input:", choice)  # 记录输入
            if choice == "1":
                temp_file = os.path.join(wav_dir, f"ranked_{i}_tone1.wav")
                generate_and_play(
                    filler_amplitude=combinations1[max_reward_index][0],
                    filler_frequency=combinations1[max_reward_index][1],
                    filler_density=combinations1[max_reward_index][2],
                    filler_env_gradient=combinations1[max_reward_index][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
                user_selections.append("Played Signal 1:" + str(combinations1[max_reward_index]))
            
            elif choice == "2":
                temp_file = os.path.join(wav_dir, f"ranked_{i}_tone2.wav")
                generate_and_play(
                    filler_amplitude=combinations1[maxinfogain_index][0],
                    filler_frequency=combinations1[maxinfogain_index][1],
                    filler_density=combinations1[maxinfogain_index][2],
                    filler_env_gradient=combinations1[maxinfogain_index][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
                user_selections.append("Played Signal 2:" + str(combinations1[maxinfogain_index]))
            
            elif choice == "3":
                selected_signal.append(1)
                user_selections.append("Selected Signal 1：" + str(combinations1[max_reward_index]))
                break
                
            elif choice == "4":
                selected_signal.append(-1)
                user_selections.append("Selected Signal 2：" + str(combinations1[maxinfogain_index]))
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # 更新 GP 参数
        gp.updateParameters([combinations1[max_reward_index], combinations1[maxinfogain_index]], selected_signal[iternum + i -1])
        # 保存 gp.updateParameters 的数据
        gp_updates.append({
            'Round': iternum + i,
            'Pair1_A': combinations1[max_reward_index][0],
            'Pair1_F': combinations1[max_reward_index][1],
            'Pair1_D': combinations1[max_reward_index][2],
            'Pair1_G': combinations1[max_reward_index][3],
            'Pair2_A': combinations1[maxinfogain_index][0],
            'Pair2_F': combinations1[maxinfogain_index][1],
            'Pair2_D': combinations1[maxinfogain_index][2],
            'Pair2_G': combinations1[maxinfogain_index][3],
            'Selection': selected_signal[iternum + i -1]
        })

    # ================================
    # 3) 最终评估阶段：用 KMeans 聚类
    # ================================
    # 原代码(如下)是从小范围抽 16 个点，这里改为用 KMeans 在 64 个原空间上做聚类:
    # dim1_vals = [33, 66]
    # dim2_vals = [33, 66]
    # dim3_vals = [20, 80]
    # dim4_vals = [-80, 80]

    amplitude = [30, 60]
    frequency = [20, 40, 60, 80]
    density = [5, 25, 50, 80]
    gradient = [-50, 50]

    
    all_combinations = list(product(amplitude, frequency, density, gradient))
    combinations_array = np.array(all_combinations)
    
    # 用 KMeans 找到 16 个代表性点
    random_signals = get_kmeans_signals(combinations_array, num_clusters=16, random_state=42)
    rsignal_ratings = np.zeros((16, 5))
    
    print("Now entering Testing Stage, ask user to take a break if necessary...")
    for i in range(16):
        print(f"Playing signal ({i+1}/16)...")

        # print("\n-----------------------------------")
        # print(f"Signal {i+1}: {random_signals[i]}")
        # print("-----------------------------------\n")

        temp_file = os.path.join(wav_dir, f"testRandom{i}.wav")
        generate_and_play(
            filler_amplitude=random_signals[i][0],
            filler_frequency=random_signals[i][1],
            filler_density=random_signals[i][2],
            filler_env_gradient=random_signals[i][3],
            duration=4,
            cycles=1,
            fs=44100,
            temp_file=temp_file
        )

        while True:
            print("Rate the pleasantness of the signal (1-7) or repeat (r): ")
            choice = input(">>> ").strip()
            print("User input:", choice)
            if choice.lower() == "r":
                print(f"Replaying signal ({i+1}/16)...")
                generate_and_play(
                    filler_amplitude=random_signals[i][0],
                    filler_frequency=random_signals[i][1],
                    filler_density=random_signals[i][2],
                    filler_env_gradient=random_signals[i][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
            elif choice in [str(x) for x in range(1, 8)]:
                rsignal_ratings[i][0] = int(choice)
                break
            else:
                print("Invalid choice. Please enter a number 1-7 or 'r' to repeat.")

        while True:
            print("Rate the urgency of the signal (1-7) or repeat (r): ")
            choice = input(">>> ").strip()
            print("User input:", choice)
            if choice.lower() == "r":
                print(f"Replaying signal ({i+1}/16)...")
                generate_and_play(
                    filler_amplitude=random_signals[i][0],
                    filler_frequency=random_signals[i][1],
                    filler_density=random_signals[i][2],
                    filler_env_gradient=random_signals[i][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
            elif choice in [str(x) for x in range(1, 8)]:
                rsignal_ratings[i][1] = int(choice)
                break
            else:
                print("Invalid choice. Please enter a number 1-7 or 'r' to repeat.")

        while True:
            print("Rate the Valence of the signal (1-9) or repeat (r): ")
            choice = input(">>> ").strip()
            print("User input:", choice)
            if choice.lower() == "r":
                print(f"Replaying signal ({i+1}/16)...")
                generate_and_play(
                    filler_amplitude=random_signals[i][0],
                    filler_frequency=random_signals[i][1],
                    filler_density=random_signals[i][2],
                    filler_env_gradient=random_signals[i][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
            elif choice in [str(x) for x in range(1, 10)]:
                rsignal_ratings[i][2] = int(choice)
                break
            else:
                print("Invalid choice. Please enter a number 1-9 or 'r' to repeat.")

        while True:
            print("Rate the Arousal of the signal (1-9) or repeat (r): ")
            choice = input(">>> ").strip()
            print("User input:", choice)
            if choice.lower() == "r":
                print(f"Replaying signal ({i+1}/16)...")
                generate_and_play(
                    filler_amplitude=random_signals[i][0],
                    filler_frequency=random_signals[i][1],
                    filler_density=random_signals[i][2],
                    filler_env_gradient=random_signals[i][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
            elif choice in [str(x) for x in range(1, 10)]:
                rsignal_ratings[i][3] = int(choice)
                break
            else:
                print("Invalid choice. Please enter a number 1-9 or 'r' to repeat.")

        while True:
            print("Rate the Dominance of the signal (1-9) or repeat (r): ")
            choice = input(">>> ").strip()
            print("User input:", choice)
            if choice.lower() == "r":
                print(f"Replaying signal ({i+1}/16)...")
                generate_and_play(
                    filler_amplitude=random_signals[i][0],
                    filler_frequency=random_signals[i][1],
                    filler_density=random_signals[i][2],
                    filler_env_gradient=random_signals[i][3],
                    duration=4,
                    cycles=1,
                    fs=44100,
                    temp_file=temp_file
                )
            elif choice in [str(x) for x in range(1, 10)]:
                rsignal_ratings[i][4] = int(choice)
                break
            else:
                print("Invalid choice. Please enter a number 1-9 or 'r' to repeat.")

    print("Thanks for participating in the user study, have a wonderful day!")
    
    # 保存所有数据
    pd.DataFrame(selected_signal).to_csv(os.path.join(output_dir, "selected_signal.csv"), index=False)
    pd.DataFrame(rsignal_ratings, columns=["Pleasantness", "Urgency", "Valence", "Arousal", "Dominance"]).to_csv(os.path.join(output_dir, "rsratings.csv"), index=False)
    pd.DataFrame(user_selections, columns=["User_Selections"]).to_csv(os.path.join(output_dir, "user_selections.csv"), index=False)
    pd.DataFrame(gp_updates).to_csv(os.path.join(output_dir, "gp_updates.csv"), index=False)  # 新增：保存 gp.updateParameters 的数据

    # 恢复 stdout 并关闭 Tee
    sys.stdout = original_stdout
    tee.close()


def get_next_filename():
    results_root = "./results"
    os.makedirs(results_root, exist_ok=True)

    current_date = time.strftime("%Y%m%d")
    subdirs = [d for d in os.listdir(results_root) 
               if os.path.isdir(os.path.join(results_root, d))]
    same_date_dirs = [d for d in subdirs if d.startswith(current_date)]
    suffixes = []
    for d in same_date_dirs:
        tail = d[len(current_date):]
        if tail.isdigit():
            suffixes.append(int(tail))
    
    if len(suffixes) == 0:
        next_num = 1
    else:
        next_num = max(suffixes) + 1

    suffix_str = f"{next_num:03d}"
    filename = current_date + suffix_str
    return filename

if __name__ == "__main__":
    auto_filename = get_next_filename()
    user_study(auto_filename)
