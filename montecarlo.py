from move_encoder import *
import math
import numpy as np
from encoder import encode_board
from move_encoder import *
import torch.nn.functional as F
from nshogi import Board

class Tree:
    def __init__(self,root):
        self.root = None

class Node:
    def __init__(self, state, parent=None):
        self.state = Board(state.sfen())
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

def softmax_temperature(x, temp=1.0):
    x = x - np.max(x)
    e_x = np.exp(x / temp)
    return e_x / e_x.sum()

class MCTS:
    def __init__(self, model, simulations=100, c_puct=2.0, device=None):
        self.model = model
        self.root = None
        self.simulations = simulations
        self.device = device if device is not None else next(model.parameters()).device
        self.c_puct = c_puct

    def run(self, board):
        self.root = Node(board)
        state_tensor = encode_board(board).unsqueeze(0).to(self.device)
        policy, val = self.model(state_tensor)
        policy = F.softmax(policy[0], dim=0)

        for move in board.legal_moves:
            index = move_to_index(move)
            if index is None or index < 0 or index >= len(policy):
                continue
            new_board = Board(board.sfen())
            new_board.push(move)
            self.root.children[move] = Node(new_board)
            self.root.children[move].prior = policy[index].item()

        for i, sim in enumerate(range(self.simulations)):
            node = self.root
            path = [node]

            while node.is_expanded():
                move, node = self.select_child(node)
                path.append(node)

            value = self.evaluate_leaf(node)
            self.backpropagate(path, value)

        visit_counts = [child.visit_count for child in self.root.children.values()]
        moves = list(self.root.children.keys())


        policy_target = np.zeros(13952)
        total_visits = sum(visit_counts)
        for move, count in zip(moves, visit_counts):
            index = move_to_index(move)
            if 0 <= index < 13952:
                policy_target[index] = count / total_visits


        return self.select_action(self.root), policy_target

    def select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_move = move
                best_child = child

        return best_move, best_child

    def evaluate_leaf(self, node):
        x = encode_board(node.state).unsqueeze(0).to(self.device)
        p, value = self.model(x)
        return value.item()

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def select_action(self, root, temperature=1.0):
        visits = torch.tensor(
            [child.visit_count for child in root.children.values()],
            dtype=torch.float32,
            device=self.device
        )

        moves = list(root.children.keys())

        if temperature == 0:
            return moves[torch.argmax(visits).item()]
        probs = softmax_temperature(visits.cpu().numpy(), temp=temperature)
        choice = torch.multinomial(torch.tensor(probs), num_samples=1).item()
        return moves[choice]
