import torch
import shogi
from network import ShogiNet
from encoder import encode_board
from montecarlo import MCTS

model_path1 = './models/model-it0.bin'

model_path2 = './models/model-it31.bin'

def play_against_model(model_path, model_path2, simulations=100):
    # Load model
    model = ShogiNet()
    model2= ShogiNet()
    model2.load_state_dict(torch.load(model_path2, map_location='cpu'))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model2.eval()

    board = shogi.Board()

    mcts = MCTS(model, simulations=simulations)
    mcts2 = MCTS(model, simulations=simulations)

    print("Play as BLACK (先手) — enter moves in USI format (e.g. 7g7f).")
    print("Model is WHITE (後手).")

    while not board.is_game_over():
        print(board)

        if board.turn == shogi.BLACK:
            # Model's turn
            print("Model2 is thinking...")
            move = mcts2.run(board)
            print(f"Model plays: {move.usi()}")
            board.push(move)
            # Human's turn
            #move_str = input("Your move (USI): ").strip()
            #try:
            #    move = shogi.Move.from_usi(move_str)
            #    if move not in board.legal_moves:
            #        raise ValueError("Illegal move")
            #    board.push(move)
            #except Exception as e:
            #    print(f"Invalid move: {e}")
            #    continue
        else:
            # Model's turn
            print("Model is thinking...")
            move = mcts.run(board)
            print(f"Model plays: {move.usi()}")
            board.push(move)

    print("Game over.")

play_against_model(model_path1,model_path2)
