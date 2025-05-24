import torch
import torch.nn.functional as F
from torch.optim import AdamW

from shogi_env import ShogiEnv
from network import ShogiNet
from montecarlo import MCTS
from encoder import encode_board
from move_encoder import *

import numpy as np
from tqdm import tqdm

def train():
    num_it = 100
    num_epoch = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for iteration in tqdm(range(num_it)):
        print(f'Running iteration {iteration + 1}:')
        env = ShogiEnv()
        env.reset()
        mcts = MCTS(model)

        data = []

        while not env.is_terminal():
            move = mcts.run(env.board)
            x = encode_board(env.board)
            env.play(move)
            data.append([x, move, 0])

        result = env.result()

        for i in range(len(data)):
            if (i % 2 == 1 and result == 1) or (i % 2 == 0 and result == -1):
                data[i][2] = 1
            else:
                data[i][2] = -1

        for epoch in range(num_epoch):
            for x, move, z in data:
                x = x.unsqueeze(0).to(device)
                pi, v = model(x)

                policy_target = torch.zeros(13952, dtype=torch.float32)

                for move, child in mcts.root.children.items():
                    index = move_to_index(move)
                    if index is not None:
                        policy_target[index] = child.visit_count

                policy_target /= policy_target.sum() + 1e-8
                target_pi = policy_target.unsqueeze(0).to(device)
                target_v = torch.tensor([[z]], dtype=torch.float32).to(device)

                loss_pi = -torch.sum(target_pi * torch.log_softmax(pi, dim=1))
                loss_v = F.mse_loss(v, target_v)
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1} loss: {loss.item():.4f}')

        torch.save(model.state_dict(), f'./models/model-it{iteration}.bin')

if __name__ == "__main__":
    train()
