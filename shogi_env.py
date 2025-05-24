import shogi

class ShogiEnv:
    def __init__(self):
        self.board = shogi.Board()

    def reset(self):
        self.board.reset()

    def is_terminal(self):
        return self.board.is_game_over()

    def result(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == shogi.BLACK else -1
        return 0  # draw or repetition

    def legal_moves(self):
        return list(self.board.legal_moves)

    def play(self, move):
        self.board.push(move)

    def get_state(self):
        return self.board.copy()

    def undo(self):
        self.board.pop()
