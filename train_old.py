import torch
import torch.nn.functional as F
from torch.optim import AdamW

from shogi_env import ShogiEnv
from encoder import encode_board
from network import ShogiNet
import shogi.KIF 
from shogi import Board

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

num_epochs = 10

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    path = './kif'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in tqdm(files):
        env = ShogiEnv()
        env.reset()
        kif = shogi.KIF.Parser.parse_file(f'{path}/{file}')[0]

     # make sure not a handicap game

        if kif['sfen'] == 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1': 
            print(kif)
            moves = kif['moves']
            win   = kif['win']
            data = []

            for i in range(len(moves)): # collect data
                move = shogi.Move.from_usi(moves[i])

                x = encode_board(env.board)
                env.play(move)

                if (win == 'b'):
                    if (i % 2 == 0):
                        data.append((x,move,1))
                    else:
                        data.append((x,move,-1))
                elif (win == 'w'):
                    if (i % 2 == 1):
                        data.append((x,move,1))
                    else:
                        data.append((x,move,-1))
                else:
                    data.append((x,move,0))
        
            for epoch in range(num_epochs):
                for x, move, z in data:
                    
                    x = x.unsqueeze(0).to(device)
                    pi, v = model(x)
                    target_pi = torch.zeros_like(pi)
                    target_v = torch.tensor([[z]], dtype=torch.float32).to(device)
                    loss = F.mse_loss(v, target_v)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Epoch {epoch + 1} loss: {loss}')

        torch.save(model.state_dict(), f'./models/kif_model/model.bin')


train()
