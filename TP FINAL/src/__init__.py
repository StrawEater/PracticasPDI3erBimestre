"""
Módulo central para el TP Final: Rompecabezas PDI.
"""

from .generator import PuzzleGenerator, create_puzzle
from .metrics import evaluate_puzzle
from .evaluator import PuzzleEvaluator
from .baseline_solver import BaselineSolver
from .affinity_matcher import compute_all_pairwise_relations, preprocess_piece, compute_border_dissimilarity
from .reconstruction import PuzzleReconstructor, reconstruct_from_relations

__all__ = [
    "PuzzleGenerator",
    "create_puzzle",
    "evaluate_puzzle",
    "PuzzleEvaluator",
    "BaselineSolver",
    "compute_all_pairwise_relations",
    "preprocess_piece",
    "compute_border_dissimilarity",
    "PuzzleReconstructor",
    "reconstruct_from_relations",
]
