import numpy as np
import shogi
import torch

def encode_board(board: shogi.Board) -> torch.Tensor:
    planes = np.zeros((119, 9, 9), dtype=np.float32)
    for sq in shogi.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            piece_type = piece.piece_type
            color = piece.color
            channel = (piece_type - 1) + (0 if color == shogi.BLACK else 7)
            row, col = divmod(sq, 9)
            planes[channel, row, col] = 1.0
    return torch.tensor(planes)
