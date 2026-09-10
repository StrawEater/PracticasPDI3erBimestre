"""
Módulo de Reconstrucción del Rompecabezas mediante Búsqueda Voraz y Backtracking.
Este módulo es completamente independiente del procesamiento de imágenes:
recibe ÚNICAMENTE las matrices/tensores de relación y las dimensiones de la grilla,
y resuelve la asignación óptima de piezas en la cuadrícula.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np


class PuzzleReconstructor:
    def __init__(
        self,
        relations: Dict[str, Any],
        rows: int,
        cols: int,
        beam_width: int = 3,
        max_backtracks: int = 500
    ):
        """
        Args:
            relations: Diccionario con afinidades 'horizontal' y 'vertical'.
            rows: Número de filas de la grilla.
            cols: Número de columnas de la grilla.
            beam_width: Cantidad máxima de mejores candidatos a explorar por celda en backtracking.
            max_backtracks: Límite máximo de retrocesos para garantizar ejecución en tiempo acotado.
        """
        self.horiz = relations["horizontal"]
        self.vert = relations["vertical"]
        self.piece_ids = relations["piece_ids"]
        self.rotations = relations["rotations"]
        self.rows = rows
        self.cols = cols
        self.beam_width = beam_width
        self.max_backtracks = max_backtracks
        
        self.backtrack_count = 0
        self.best_grid = None
        self.best_rotations = None
        self.min_total_cost = float("inf")

    def _get_compatibility_cost(
        self,
        grid: List[List[int]],
        rotations: Dict[int, int],
        r: int,
        c: int,
        candidate_p: int,
        candidate_rot: int
    ) -> float:
        """
        Calcula el costo medio de colocar (candidate_p, candidate_rot) en (r, c)
        respecto a todos los vecinos ya colocados.
        """
        cost = 0.0
        neighbors_count = 0
        
        # Vecino Izquierdo (c - 1): Izquierdo está a la izquierda de candidato
        if c > 0 and grid[r][c - 1] != -1:
            left_p = grid[r][c - 1]
            left_rot = rotations[left_p]
            key = (left_p, left_rot, candidate_p, candidate_rot)
            cost += self.horiz.get(key, 1e6)
            neighbors_count += 1
            
        # Vecino Derecho (c + 1): Candidato está a la izquierda del derecho
        if c + 1 < self.cols and grid[r][c + 1] != -1:
            right_p = grid[r][c + 1]
            right_rot = rotations[right_p]
            key = (candidate_p, candidate_rot, right_p, right_rot)
            cost += self.horiz.get(key, 1e6)
            neighbors_count += 1
            
        # Vecino Superior (r - 1): Superior está arriba de candidato
        if r > 0 and grid[r - 1][c] != -1:
            top_p = grid[r - 1][c]
            top_rot = rotations[top_p]
            key = (top_p, top_rot, candidate_p, candidate_rot)
            cost += self.vert.get(key, 1e6)
            neighbors_count += 1
            
        # Vecino Inferior (r + 1): Candidato está arriba del inferior
        if r + 1 < self.rows and grid[r + 1][c] != -1:
            bottom_p = grid[r + 1][c]
            bottom_rot = rotations[bottom_p]
            key = (candidate_p, candidate_rot, bottom_p, bottom_rot)
            cost += self.vert.get(key, 1e6)
            neighbors_count += 1
            
        return (cost / neighbors_count) if neighbors_count > 0 else 0.0

    def _select_most_constrained_cell(
        self,
        grid: List[List[int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Heurística MRV (Most Constrained Variable):
        Selecciona la celda vacía adyacente a la mayor cantidad de vecinos ya colocados.
        Esto reduce drásticamente el espacio de búsqueda.
        """
        best_cell = None
        max_neighbors = -1
        
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == -1:
                    # Contar cuántos vecinos ya están colocados
                    neighbors = 0
                    if c > 0 and grid[r][c - 1] != -1: neighbors += 1
                    if c + 1 < self.cols and grid[r][c + 1] != -1: neighbors += 1
                    if r > 0 and grid[r - 1][c] != -1: neighbors += 1
                    if r + 1 < self.rows and grid[r + 1][c] != -1: neighbors += 1
                    
                    if neighbors > max_neighbors:
                        max_neighbors = neighbors
                        best_cell = (r, c)
                        
        return best_cell

    def _find_best_seed_pair(self) -> Tuple[int, int, int, int]:
        """
        Encuentra el par de piezas adyacentes con menor disimilitud global
        para iniciar el rompecabezas con máxima confianza.
        Retorna: (p_a, rot_a, p_b, rot_b) en relación horizontal.
        """
        best_key = min(self.horiz.keys(), key=lambda k: self.horiz[k])
        return best_key

    def _backtrack_search(
        self,
        grid: List[List[int]],
        rotations: Dict[int, int],
        used_pieces: Set[int],
        accumulated_cost: float
    ) -> bool:
        """
        Búsqueda recursiva eficiente Greedy + Backtracking.
        """
        # Si todas las piezas fueron asignadas, evaluamos la solución completa
        if len(used_pieces) == len(self.piece_ids):
            if accumulated_cost < self.min_total_cost:
                self.min_total_cost = accumulated_cost
                self.best_grid = [row.copy() for row in grid]
                self.best_rotations = rotations.copy()
            return True

        # Poda por límite de retrocesos
        if self.backtrack_count >= self.max_backtracks:
            return False

        # 1. Seleccionar la celda más restringida
        cell = self._select_most_constrained_cell(grid)
        if cell is None:
            return False
        r, c = cell

        # 2. Ordenar candidatos según costo de compatibilidad
        candidate_costs = []
        for candidate_p in self.piece_ids:
            if candidate_p in used_pieces:
                continue
            for rot in self.rotations:
                cost = self._get_compatibility_cost(grid, rotations, r, c, candidate_p, rot)
                candidate_costs.append((cost, candidate_p, rot))

        candidate_costs.sort(key=lambda x: x[0])
        # Seleccionamos hasta beam_width mejores opciones para mantener eficiencia
        top_candidates = candidate_costs[:self.beam_width]

        # 3. Probar cada candidato (Greedy con retroceso)
        for cost, p, rot in top_candidates:
            # Colocar
            grid[r][c] = p
            rotations[p] = rot
            used_pieces.add(p)

            # Avanzar recursivamente
            success = self._backtrack_search(grid, rotations, used_pieces, accumulated_cost + cost)
            if success and self.min_total_cost == 0:
                return True # Solución perfecta encontrada

            # Deshacer (Backtrack)
            grid[r][c] = -1
            del rotations[p]
            used_pieces.remove(p)
            self.backtrack_count += 1

            if self.backtrack_count >= self.max_backtracks:
                break

        return False

    def reconstruct(self, anchor: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        Ejecuta la reconstrucción completa.
        
        Args:
            anchor: Tupla opcional (piece_id, row, col, rot) para fijar una pieza ancla (e.g. marcador).
        Retorna:
            (grid_resuelto, rotaciones_resueltas)
        """
        grid = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]
        rotations = {}
        used_pieces = set()

        initial_cost = 0.0
        if anchor is not None:
            anc_p, anc_r, anc_c, anc_rot = anchor
            grid[anc_r][anc_c] = anc_p
            rotations[anc_p] = anc_rot
            used_pieces.add(anc_p)
        else:
            # 1. Semilla inicial: Colocar el par más compatible en (0, 0) y (0, 1) si cols > 1
            p_a, rot_a, p_b, rot_b = self._find_best_seed_pair()
            
            grid[0][0] = p_a
            rotations[p_a] = rot_a
            used_pieces.add(p_a)

            if self.cols > 1:
                grid[0][1] = p_b
                rotations[p_b] = rot_b
                used_pieces.add(p_b)
                initial_cost = self.horiz.get((p_a, rot_a, p_b, rot_b), 0.0)

        # Guardar como base preliminar
        self.best_grid = [row.copy() for row in grid]
        self.best_rotations = rotations.copy()

        # 2. Ejecutar búsqueda con backtracking
        self._backtrack_search(grid, rotations, used_pieces, initial_cost)

        # Si el backtracking no completó por límite de iteraciones, rellenar de forma greedy pura el remanente
        self._fill_remaining_greedy(self.best_grid, self.best_rotations)
        return self.best_grid, self.best_rotations

    def _fill_remaining_greedy(self, grid: List[List[int]], rotations: Dict[int, int]) -> None:
        """Relleno de contingencia en caso de que el límite de backtracks se alcance."""
        placed = set(rotations.keys())
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == -1:
                    best_cost = float("inf")
                    best_choice = None
                    best_rot = 0
                    for p in self.piece_ids:
                        if p in placed:
                            continue
                        for rot in self.rotations:
                            cost = self._get_compatibility_cost(grid, rotations, r, c, p, rot)
                            if cost < best_cost:
                                best_cost = cost
                                best_choice = p
                                best_rot = rot
                                
                    if best_choice is not None:
                        grid[r][c] = best_choice
                        rotations[best_choice] = best_rot
                        placed.add(best_choice)


def reconstruct_from_relations(
    relations: Dict[str, Any],
    rows: int,
    cols: int,
    beam_width: int = 3,
    max_backtracks: int = 500,
    anchor: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Función principal de reconstrucción basada en relaciones de afinidad.
    """
    reconstructor = PuzzleReconstructor(
        relations=relations,
        rows=rows,
        cols=cols,
        beam_width=beam_width,
        max_backtracks=max_backtracks
    )
    return reconstructor.reconstruct(anchor=anchor)
