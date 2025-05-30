import torch
import traceback
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from network import ShogiNet
from montecarlo import MCTS
from encoder import encode_board
from move_encoder import move_to_index
from dataset import ShogiDataset
from checkpoints import save_checkpoint, load_checkpoint
from shogi_env import ShogiEnv

import os
import threading
import time
import datetime
from copy import deepcopy

import numpy as np
from tqdm import tqdm, trange

#set variables
num_games = 10
num_iter  = 10
num_epochs= 10
num_workers = 2

def train_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    model_path = './checkpoints/latest_model.pt'
    checkpoint_path = f'./checkpoints/model-rank{rank}.pt'

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model = ShogiNet().to(device)



    if rank == 0 and not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)




    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")

    ddp_model = DDP(model, device_ids=[rank])

    dist.barrier()

    optimizer = AdamW(ddp_model.parameters(), lr=1e-3)
    start_iter = load_checkpoint(model, optimizer, path=checkpoint_path, map_location=device)


    for iteration in range(start_iter, num_iter):
        if rank == 0:
            print(f'Running iteration {iteration + 1}')
        ddp_model.eval()

        if rank == 0:
            print(f'generating for iter {iteration}')
            samples = play_games(model, num_games // world_size, device)
            print(f'done generating for iter {iteration}')
        else: 
            samples = None

        samples_list = [samples]  # broadcast_object_list requires a list
        dist.broadcast_object_list(samples_list, src=0)

        ddp_model.train()

        dataset = ShogiDataset(samples)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

        for i, epoch in enumerate(range(num_epochs)):
            sampler.set_epoch(epoch)
            for x, pi, v in dataloader:
                x, pi, v = x.to(device), pi.to(device), v.to(device)
                pred_pi, pred_v = ddp_model(x)
                log_probs = F.log_softmax(pred_pi, dim=1)

                loss_pi = F.kl_div(log_probs, pi, reduction='batchmean')
                loss_v = F.mse_loss(pred_v, v)
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'loss of epoch {i} is {loss.item():.4f}')

        if rank == 0:
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            torch.save(model.state_dict(), "./checkpoints/latest_model.pt")

    dist.destroy_process_group()



def self_play(model, device):
    env = ShogiEnv()
    env.reset()
    mcts = MCTS(model, simulations=50, device=device)
    data = []

    while not env.is_terminal():
        move, policy_target = mcts.run(env.board)
        x = encode_board(env.board)
        env.play(move)
        data.append([x, policy_target.copy(), 0])

    result = env.result()
    for i in range(len(data)):
        data[i][2] = 1 if (i % 2 == 0 and result == -1) or (i % 2 == 1 and result == 1) else -1 if result != 0 else 0


    samples = [(x, policy_target.copy(), z) for x, policy_target, z in data]

    return samples



def generate_games_batch(device, queue, num_games, model_path, counter, lock):
    try:
        model = ShogiNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        scripted_model = torch.jit.script(model)

        data = []
        with torch.no_grad():
            for _ in range(num_games):
                data.extend(self_play(scripted_model, device))
                with lock:
                    counter.value += 1
        print(f'length of data is {len(data)}')
        queue.put(data)
        torch.cuda.empty_cache()
        print(f"[Worker PID {os.getpid()}] Finished with {len(data)} samples")
    except Exception as e:
        print(f'length of data is {len(data)}')
        print(f"[PID {os.getpid()}] Worker failed: {e}")
        queue.put([])  # Ensure queue.get() doesn’t hang


def play_games(model, num_games, device, model_path="./checkpoints/latest_model.pt"):
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file at: {model_path}")
    ctx = mp.get_context('spawn')

    manager = ctx.Manager()
    queue = manager.Queue()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    total_games = num_games * num_workers

    progress_bar = tqdm(total=num_games, desc="Generating games")

    def monitor_progress():
        last = 0
        while progress_bar.n < num_games:
            with lock:
                current = counter.value
            progress_bar.update(current - last)
            last = current
            time.sleep(0.1)
        progress_bar.close()

    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.start()

    games_per_worker = num_games // num_workers
    


    
    all_data = []
    procs = []
    for _ in range(num_workers):
        print("starting process")
        p = ctx.Process(target=generate_games_batch, args=(device, queue, games_per_worker, model_path, counter, lock))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("done starting processes")
    for i in range(num_workers): # FIXME gets file not found or timeout error here
        try:
            print(f'size of queue is: {queue.qsize()}')
            data = queue.get(timeout=30)  # wait max 5 minutes
            all_data.extend(data)
        except Exception as e:
            print(f"Worker failed or timed out: {e}")



    return all_data

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiNet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for iteration in range(num_iter):
        print(f'Running iteration {iteration + 1}')
        model.eval()
        model.train()
        samples = play_games(model, 100 // world_size, device)

        dataset = ShogiDataset(samples)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



        for epoch in range(num_epochs):
            pred_list = []
            target_list = []
            for x_, pi_, v_ in dataloader:
                pred_list.append(model.predict_on_batch(x_))
                target_list.append(deepcopy((pi_,v_)))
            for pred, target, in zip(pred_list,target_list):
                pred_pi, pred_v = pred
                target_pi, target_v = target
                loss_pi = -torch.sum(target_pi * torch.log_softmax(pred_pi, dim=1)) / x.size(0)
                loss_v = F.mse_loss(pred_v, target_v)
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1} loss: {loss.item():.4f}')

        torch.save(model.state_dict(), f'./models/model-it{iteration}.pt')

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    #train() 

