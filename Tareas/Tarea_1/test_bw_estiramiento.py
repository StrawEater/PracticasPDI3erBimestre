import unittest
import numpy as np
import sys

funcion_bw = None
funcion_dist = None
funcion_uniforme = None
funcion_matriz = None
funcion_ecualizar = None

class TestBWEstiramiento(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gradiente = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
        cls.plana = np.full((32, 32), 128, dtype=np.uint8)
        np.random.seed(42)
        cls.aleatoria = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)

    def test_mismo_tamano(self):
        for img in [self.gradiente, self.plana, self.aleatoria]:
            for lam in [0, 1]:
                for alpha in [0, 1]:
                    for b, w in [(0, 255), (50, 200)]:
                        eq, _ = funcion_bw(img, lam, alpha, b, w)
                        self.assertEqual(eq.shape, img.shape,
                                         f"Tamaño incorrecto para λ={lam}, α={alpha}, b={b}, w={w}")

    def test_rango_correcto(self):
        for img in [self.gradiente, self.plana, self.aleatoria]:
            for lam in [0, 0.5]:
                for alpha in [0, 0.5]:
                    for b, w in [(0, 255), (30, 220)]:
                        eq, _ = funcion_bw(img, lam, alpha, b, w)
                        self.assertTrue(np.all(eq >= 0) and np.all(eq <= 255),
                                         f"Valores fuera de rango para λ={lam}, α={alpha}, b={b}, w={w}")
                        self.assertEqual(eq.dtype, np.uint8,
                                         f"Tipo incorrecto: {eq.dtype}")

    def test_no_modifica_original(self):
        img = self.aleatoria.copy()
        copia = img.copy()
        _, _ = funcion_bw(img, 1.0, 1.0, 50, 200)
        np.testing.assert_array_equal(img, copia,
                                      "La función modificó la imagen original (falta .copy())")

    def test_lambda_alpha_cero(self):
        """λ=0, α=0 → h = hi, ecualización estándar."""
        img = self.aleatoria
        hi = funcion_dist(img)
        eq, h = funcion_bw(img, 0.0, 0.0, 0, 255)
        #h debe ser igual a hi (tolerancia numérica)
        np.testing.assert_allclose(h, hi, rtol=1e-6, atol=1e-6,
                                   err_msg="h no coincide con hi para λ=0, α=0")
        #Imagen ecualizada estándar
        eq_std = funcion_ecualizar(img, hi)
        np.testing.assert_array_equal(eq, eq_std,
                                      "Imagen no coincide con ecualización estándar")


    def test_residuo_sistema(self):
        """Verifica que ( (1+λ)I + α I^B ) h ≈ h0 + λ u."""
        img = self.aleatoria
        h0 = funcion_dist(img)
        u = funcion_uniforme()
        I = np.eye(256)
        for lam in [0, 1, 5]:
            for alpha in [0, 1, 5]:
                for b, w in [(0, 255), (30, 200)]:
                    IB = funcion_matriz(b, w)
                    A = (1 + lam) * I + alpha * IB
                    b_vec = h0 + lam * u
                    _, h = funcion_bw(img, lam, alpha, b, w)
                    residual = A @ h - b_vec
                    norm_res = np.linalg.norm(residual)
                    self.assertLess(norm_res, 1e-8,
                                    f"Residuo muy grande para λ={lam}, α={alpha}, b={b}, w={w}: {norm_res}")

    def test_efecto_alpha(self):
        """Al aumentar α, la energía de h en los extremos debería disminuir."""
        img = self.aleatoria
        b, w = 50, 200
        # α=0 y α=10
        _, h0 = funcion_bw(img, 0.0, 0.0, b, w)
        _, h1 = funcion_bw(img, 0.0, 10.0, b, w)
        # a matriz I^B tiene 1 en los extremos (≤b y ≥w)
        IB = funcion_matriz(b, w)
        energia0 = h0 @ IB @ h0   # h^T I^B h
        energia1 = h1 @ IB @ h1
        # Con α mayor, la energía penalizada debería ser menor
        self.assertLess(energia1, energia0 + 1e-6,
                        f"Con α=10, energía en extremos debería ser menor que con α=0. "
                        f"Energía0={energia0}, Energía1={energia1}")

def evaluar(func_bw, func_dist, func_unif, func_mat, func_ecual):
    global funcion_bw, funcion_dist, funcion_uniforme, funcion_matriz, funcion_ecualizar
    funcion_bw = func_bw
    funcion_dist = func_dist
    funcion_uniforme = func_unif
    funcion_matriz = func_mat
    funcion_ecualizar = func_ecual

    sys.argv = ['']
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBWEstiramiento)
    runner = unittest.TextTestRunner(verbosity=0)
    resultado = runner.run(suite)

    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS: BLACK & WHITE STRETCHING")
    print("="*70)
    if resultado.wasSuccessful():
        print("¡Todas las pruebas pasaron! La función BW es correcta.")
    else:
        print("Algunas pruebas fallaron. Revisa los mensajes de error arriba.")
        print(f"Fallos: {len(resultado.failures)}")
        print(f"Errores: {len(resultado.errors)}")