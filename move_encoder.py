import shogi
import torch

NUM_SQUARES = 81
NUM_MOVE_INDICES = 13952
DROP_PIECE_TYPES = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]

DIRECTION_OFFSETS = [
    (-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0), (-8, 0),  # Up
    (-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7), (-8, 8),  # Up-right
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),          # Right
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),          # Down-right
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),          # Down
    (1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7), (8, -8),  # Down-left
    (0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7), (0, -8),  # Left
    (-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7), (-8, -8),  # Up-left
    (-2, 1), (-2, -1), (2, 1), (2, -1),  # Knight jumps
]

# ------- Move to Index -------

def move_to_index(move: shogi.Move) -> int:
    if move.drop_piece_type is not None:
        pt_index = DROP_PIECE_TYPES.index(move.drop_piece_type)
        return 13392 + pt_index * NUM_SQUARES + move.to_square
    else:
        from_sq = move.from_square
        to_sq = move.to_square
        from_row, from_col = divmod(from_sq, 9)
        to_row, to_col = divmod(to_sq, 9)
        dr, dc = to_row - from_row, to_col - from_col

        if (dr, dc) in [(-2, 1), (-2, -1), (2, 1), (2, -1)]:
            offset = DIRECTION_OFFSETS.index((dr, dc))
        else:
            if dr != 0: dr //= abs(dr)
            if dc != 0: dc //= abs(dc)
            try:
                offset = DIRECTION_OFFSETS.index((dr, dc))
            except ValueError:
                return -1

        promo = 1 if move.promotion else 0
        return (offset * 2 + promo) * NUM_SQUARES + from_sq

def move_to_index_tensor(move: shogi.Move) -> torch.Tensor:
    idx = move_to_index(move)
    tensor = torch.zeros(NUM_MOVE_INDICES, dtype=torch.float32)
    if idx != -1:
        tensor[idx] = 1.0
    return tensor

# ------- Index to Move -------

def index_to_move(index: int, board: shogi.Board) -> shogi.Move:
    if index < 0 or index >= NUM_MOVE_INDICES:
        return None

    if index >= 13392:
        drop_index = index - 13392
        pt_index = drop_index // NUM_SQUARES
        to_sq = drop_index % NUM_SQUARES
        return shogi.Move(None, to_sq, drop_piece_type=DROP_PIECE_TYPES[pt_index])
    else:
        from_sq = index % NUM_SQUARES
        offset_promo = index // NUM_SQUARES
        offset = offset_promo // 2
        promo = offset_promo % 2

        if offset >= len(DIRECTION_OFFSETS):
            return None

        dr, dc = DIRECTION_OFFSETS[offset]
        from_row, from_col = divmod(from_sq, 9)
        to_row = from_row + dr
        to_col = from_col + dc

        if not (0 <= to_row < 9 and 0 <= to_col < 9):
            return None

        to_sq = to_row * 9 + to_col
        move = shogi.Move(from_sq, to_sq, promotion=bool(promo))

        if move in board.legal_moves:
            return move
        else:
            move.promotion = not move.promotion
            return move if move in board.legal_moves else None

def index_to_move_tensor(logits: torch.Tensor, board: shogi.Board) -> shogi.Move:
    sorted_indices = torch.argsort(logits, descending=True)
    for idx in sorted_indices.tolist():
        move = index_to_move(idx, board)
        if move is not None and move in board.legal_moves:
            return move
    return None  # fallback if no legal move found
