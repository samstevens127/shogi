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

def play_games(model, num_games):
    model.share_memory()

    data = []

    for _ in range(num_games):
        self_play(model,data)

    return data


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for iteration in tqdm(range(150)):
        print(f'Running iteration {iteration + 1}')
        model.eval()
        model.train()
        samples = play_games(model, 100)

        dataset = ShogiDataset(samples)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(5):
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
