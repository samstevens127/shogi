import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from network import ShogiNet
from montecarlo import MCTS
from encoder import encode_board
from move_encoder import move_to_index
from dataset import ShogiDataset

from shogi_env import ShogiEnv
import numpy as np
from tqdm import tqdm

num_proc  = 4
num_games = 100
num_iter  = 150
num_epochs= 6

def self_play(model, dat):
    env = ShogiEnv()
    env.reset()
    mcts = MCTS(model, simulations=50)
    data = []

    while not env.is_terminal():
        move = mcts.run(env.board)
        x = encode_board(env.board)
        env.play(move)
        data.append([x, move, 0])

    result = env.result()
    for i in range(len(data)):
        data[i][2] = 1 if (i % 2 == 0 and result == -1) or (i % 2 == 1 and result == 1) else -1 if result != 0 else 0

    visit_counts = [child.visit_count for child in mcts.root.children.values()]
    moves = list(mcts.root.children.keys())
    policy_target = np.zeros(13952)
    total_visits = sum(visit_counts)
    for move, count in zip(moves, visit_counts):
        index = move_to_index(move)
        if 0 <= index < 13952:
            policy_target[index] = count / total_visits

    samples = [(x, policy_target.copy(), z) for x, _, z in data]
    dat.extend(samples)

def self_play_worker(model, queue):
    env = ShogiEnv()
    env.reset()
    mcts = MCTS(model, simulations=50)
    data = []

    print("simulating game")
    while not env.is_terminal():
        move = mcts.run(env.board)
        x = encode_board(env.board)
        env.play(move)
        data.append([x, move, 0])

    result = env.result()
    for i in range(len(data)):
        data[i][2] = 1 if (i % 2 == 0 and result == -1) or (i % 2 == 1 and result == 1) else -1 if result != 0 else 0

    visit_counts = [child.visit_count for child in mcts.root.children.values()]
    moves = list(mcts.root.children.keys())
    policy_target = np.zeros(13952)
    total_visits = sum(visit_counts)
    for move, count in zip(moves, visit_counts):
        index = move_to_index(move)
        if 0 <= index < 13952:
            policy_target[index] = count / total_visits

    samples = [(x, policy_target.copy(), z) for x, _, z in data]
    queue.put(samples)


def play_games(model, num_games):
    games_per_proc = num_games // num_proc
    model.share_memory()

    ctx = mp.get_context('spawn')  # safer for PyTorch models
    queue = ctx.Queue()
    processes = []

    for rank in range(num_proc):
        p = ctx.Process(target=run_multiple_games, args=(model, queue, games_per_proc), name=f'process-{rank}')
        p.start()
        processes.append(p)
        print(f'Started {p.name}')

    all_data = []
    for _ in range(num_proc):
        samples = queue.get()
        all_data.extend(samples)

    for p in processes:
        p.join()

    return all_data


def run_multiple_games(model, queue, num_games):
    all_samples = []
    for _ in range(num_games):
        self_play_worker(model, queue=FakeQueue(all_samples))  # Reuse single list for multiple games
    queue.put(all_samples)


class FakeQueue:
    """A simple interface to pass a list like a queue internally in worker."""
    def __init__(self, storage):
        self.storage = storage

    def put(self, item):
        self.storage.extend(item)



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for iteration in range(num_iter):
        print(f'Running iteration {iteration + 1}')
        model.eval()
        model.train()
        samples = play_games(model, num_games)

        dataset = ShogiDataset(samples)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(num_epochs):
            for x, pi, v in dataloader:
                x, pi, v = x.to(device), pi.to(device), v.to(device)

                pred_pi, pred_v = model(x)
                loss_pi = -torch.sum(pi * torch.log_softmax(pred_pi, dim=1)) / x.size(0)
                loss_v = F.mse_loss(pred_v, v)
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1} loss: {loss.item():.4f}')

        torch.save(model.state_dict(), f'./models/model-it{iteration}.bin')

if __name__ == "__main__":
    train()
