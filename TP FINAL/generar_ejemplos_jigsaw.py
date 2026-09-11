"""
Script para generar el conjunto de ejemplos oficiales del TP Final
utilizando el método de piezas de rompecabezas con encastres específicos (Macho y Hembra sobre fondo negro).
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.generator import create_puzzle


def generar_todos_los_ejemplos():
    repo_root = os.path.dirname(CURRENT_DIR)
    img_dir = os.path.join(repo_root, "imagenes")
    dataset_dir = os.path.join(CURRENT_DIR, "dataset_ejemplos")
    
    print("=" * 75)
    print("  GENERANDO DATASET DE EJEMPLOS CON ENCASTRES (MACHO/HEMBRA Y FONDO NEGRO)  ")
    print("=" * 75)
    
    ejemplos = [
        {
            "nombre": "puzzle_2x2_intro",
            "imagen": os.path.join(img_dir, "casita.jpg"),
            "rows": 2, "cols": 2,
            "rotate": False,
            "seed": 10,
            "desc": "2x2 Introductorio (4 piezas, encastres macho/hembra, ideal para depuración inicial)"
        },
        {
            "nombre": "puzzle_3x3_facil",
            "imagen": os.path.join(img_dir, "casita.jpg"),
            "rows": 3, "cols": 3,
            "rotate": False,
            "seed": 42,
            "desc": "3x3 Nivel Fácil (9 piezas, encastres macho/hembra, sin rotación)"
        },
        {
            "nombre": "puzzle_3x3_rotado",
            "imagen": os.path.join(img_dir, "cueva.jpg"),
            "rows": 3, "cols": 3,
            "rotate": True,
            "seed": 777,
            "desc": "3x3 Nivel Avanzado con Rotaciones (9 piezas con rotación ortogonal aleatoria 0/90/180/270)"
        },
        {
            "nombre": "puzzle_4x4_medio",
            "imagen": os.path.join(img_dir, "barco.jpg"),
            "rows": 4, "cols": 4,
            "rotate": False,
            "seed": 123,
            "desc": "4x4 Nivel Medio (16 piezas, encastres macho/hembra, sin rotación)"
        },
        {
            "nombre": "puzzle_10x10_desafio",
            "imagen": os.path.join(img_dir, "casita.jpg"),
            "rows": 10, "cols": 10,
            "rotate": False,
            "seed": 42,
            "desc": "10x10 Desafío Masivo (100 piezas, encastres macho/hembra de alta resolución)"
        }
    ]
    
    for ej in ejemplos:
        out_path = os.path.join(dataset_dir, ej["nombre"])
        print(f"\n>> Generando: {ej['nombre']}")
        print(f"   Descripción: {ej['desc']}")
        print(f"   Imagen: {os.path.basename(ej['imagen'])} | Dimensiones: {ej['rows']}x{ej['cols']}")
        
        create_puzzle(
            image_path=ej["imagen"],
            output_dir=out_path,
            rows=ej["rows"],
            cols=ej["cols"],
            cut_type="jigsaw",
            allow_rotations=ej.get("rotate", False),
            slight_rotation=ej.get("slight_rotation", False),
            max_jitter_degrees=12.0,
            add_stripes=ej.get("add_stripes", False),
            stripe_period=8,
            stripe_amplitude=0.32,
            add_marker=ej.get("add_marker", False),
            seed=ej["seed"]
        )
        
    print("\n" + "=" * 75)
    print("  [ÉXITO] Todos los ejemplos con encastres macho/hembra fueron generados.")
    print("=" * 75)


if __name__ == "__main__":
    generar_todos_los_ejemplos()
