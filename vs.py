import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')
import pickle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'로 변경해 볼 수 있습니다.
import folium
import imageio
import os

import folium
import imageio
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


def plot_path(all_loc, coordinates, rew, max_length, prize, title="Agent's Path(EXPONENTIAL), graph size: 50 ", gif_filename='agent_path_ex_unif_50.gif'):
    """
    Plots the path of the agent based on the visited coordinates and saves it as a GIF with prize values.

    :param all_loc: List of (x, y) tuples representing all locations.
    :param coordinates: List of (x, y) tuples representing the visited coordinates.
    :param prize: List of prize values corresponding to all locations.
    :param max_length: Maximum path length (displayed in title).
    :param title: Title of the plot.
    :param gif_filename: Filename for the output GIF.
    """

    x_min = -74.202922
    y_min = 40.5231587
    x_max = -73.696815
    y_max = 41.000784

    # 좌표 리스트에서 x, y 값을 추출
    x, y = zip(*coordinates)
    all_x, all_y = zip(*all_loc)
    print(len(all_x), len(all_y))

    # 좌표 리스트에서 x, y 값을 추출
    x, y = zip(*coordinates)

    # 정규화된 좌표를 원래 좌표로 복원
    x = [xi * (x_max - x_min) + x_min for xi in x]
    y = [yi * (y_max - y_min) + y_min for yi in y]

    all_x = [xi * (x_max - x_min) + x_min for xi in all_x]
    all_y = [yi * (y_max - y_min) + y_min for yi in all_y]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print([x,y])
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Path Coordinates in Order:")
    for xi, yi in zip(x, y):
        print(f"({yi} {xi})")

    total_rew = sum(rew)

    fig, ax = plt.subplots(figsize=(14, 8))

    # 전체 경로와 Depot을 배경으로 그리기
    ax.plot(all_x, all_y, marker='o', linestyle='', color='g', markersize=5, label='All Coordinates')
    ax.plot(x[0], y[0], marker='o', linestyle='', color='r', markersize=10, label='Depot')

    # 애니메이션을 위한 초기 설정
    line, = ax.plot([], [], marker='o', linestyle='-', color='b', markersize=5)

    # 각 좌표에 prize 값 표시
    annotations = [ax.annotate(f"{p:.2f}", (lx, ly), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='black')
                   for lx, ly, p in zip(all_x, all_y, prize)]

    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_title(title + ", total_reward: " + str(total_rew*100) + ", max_length:" + str(max_length))
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid()
    ax.axis('equal')

    # 애니메이션 업데이트 함수
    def update(frame):
        line.set_data(x[:frame + 1], y[:frame + 1])
        return line,

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=len(coordinates), blit=True)

    # GIF로 저장
    ani.save(gif_filename, writer='imagemagick', fps=10)

    plt.show()


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args, all_loc, depot, prize, max_length):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device, all_loc, depot, prize, max_length)


def eval_dataset(dataset_path, width, softmax_temp, opts, all_loc, depot, prize, max_length):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, all_loc, depot, prize, max_length, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device, all_loc, depot, prize, max_length)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations, coord = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
    print("The end")
    print("coord: ", coord)

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device, all_loc, depot, prize, max_length):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    # 좌표를 저장할 리스트 추가
    # 모든 좌표를 dataset에서 가져옴
    #print(dataset)
    # for i in range(len(dataset)):
    #     #dataset의 i번째 인스턴스에서 좌표를 가져옴
    #     coord = dataset[i][1]# dataset의 구조에 따라 조정 필요
    #     all_coord.append(coord)

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)
        print("batch: ", batch)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                print("sequences: ", sequences)
                print("costs: ", costs)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq) # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
                # 경로에 대한 좌표를 가져옵니다.
            print("seq: ", seq)
            coordinates, rew = get_coordinates_from_sequence(all_loc, depot, seq, prize)

            results.append((cost, seq, duration))  # 결과에 좌표 추가

            # 방문한 경로를 플롯합니다.
            plot_path(all_loc, coordinates, rew, max_length, prize)


            #results.append((cost, seq, duration))

    return results

def get_coordinates_from_sequence(all_coord, depot, sequence, prize):
    """
    Given a sequence of indices, return the corresponding coordinates.

    :param sequence: List of indices representing the path taken.
    :return: List of (x, y) tuples corresponding to the coordinates of the indices.
    """
    # Assuming you have a list of coordinates defined somewhere
    # For example:
    # coordinates = [(x1, y1), (x2, y2), ..., (xn, yn)]
    coordinates = []
    rew = []
    coordinates.append(depot)# Replace this with actual coordinates
    rew.append(0)
    for index in sequence:
        coordinates.append(all_coord[index - 1])# Get the coordinate for each index in the sequence
        rew.append(prize[index - 1])
    coordinates.append(depot)
    rew.append(0)
    print(coordinates)
    print(rew)
    return coordinates, rew


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:

            with open(dataset_path, 'rb') as file:
                data = pickle.load(file)

            depot = data[0][0]
            all_loc = data[0][1]
            print(len(all_loc))
            prize = data[0][2]
            print(len(prize))
            max_length = data[0][3]

            eval_dataset(dataset_path, width, opts.softmax_temperature, opts, all_loc, depot, prize, max_length)