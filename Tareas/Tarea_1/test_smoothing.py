import unittest
import numpy as np
import sys

funcion_suavizar = None
funcion_distribucion = None
funcion_uniforme = None
funcion_matrizD = None
funcion_ecualizar = None

class TestSuavizarHistograma(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gradiente = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
        cls.plana = np.full((32, 32), 128, dtype=np.uint8)
        np.random.seed(42)
        cls.aleatoria = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)

    def test_mismo_tamano(self):
        for img in [self.gradiente, self.plana, self.aleatoria]:
            for lam in [0, 1, 10]:
                for gamma in [0, 1, 10]:
                    eq, _ = funcion_suavizar(img, lam, gamma)
                    self.assertEqual(eq.shape, img.shape,
                                     f"Tamaño incorrecto para λ={lam}, γ={gamma}")

    def test_rango_correcto(self):
        for img in [self.gradiente, self.plana, self.aleatoria]:
            for lam in [0, 0.5, 5]:
                for gamma in [0, 0.5, 5]:
                    eq, _ = funcion_suavizar(img, lam, gamma)
                    self.assertTrue(np.all(eq >= 0) and np.all(eq <= 255),
                                    f"Valores fuera de rango para λ={lam}, γ={gamma}")
                    self.assertEqual(eq.dtype, np.uint8,
                                     f"Tipo incorrecto para λ={lam}, γ={gamma}: {eq.dtype}")

    def test_lambda_gamma_cero(self):
        """λ=0, γ=0 → h = hi (histograma original) y ecualización estándar."""
        img = self.aleatoria
        hi = funcion_distribucion(img)
        eq, h = funcion_suavizar(img, 0.0, 0.0)
        np.testing.assert_allclose(h, hi, rtol=1e-6, atol=1e-6,
                                   err_msg="h no coincide con hi para λ=0, γ=0")
        eq_std = funcion_ecualizar(img, hi)
        np.testing.assert_array_equal(eq, eq_std,
                                      "Imagen ecualizada no coincide con la estándar para λ=0, γ=0")

    def test_suavidad_con_gamma_alto(self):
        """Con γ muy grande, la solución debe ser suave: D h ≈ 0."""
        img = self.aleatoria
        gamma = 1000
        lam = 0
        _, h = funcion_suavizar(img, lam, gamma)
        D = funcion_matrizD()
        Dh = D @ h
        norma_Dh = np.linalg.norm(Dh)
        self.assertLess(norma_Dh, 1e-2,
                        f"Con γ={gamma}, D h no es suficientemente suave, norma={norma_Dh}")

    def test_suma_aproximada_1(self):
        """h debe sumar aproximadamente 1 (tolerancia 1e-2)."""
        img = self.aleatoria
        for lam in [0, 1, 10]:
            for gamma in [0, 1, 10]:
                _, h = funcion_suavizar(img, lam, gamma)
                suma = np.sum(h)
                self.assertAlmostEqual(suma, 1.0, delta=1e-2,
                                       msg=f"Suma de h = {suma} para λ={lam}, γ={gamma}")

    def test_no_negativo(self):
        """Todos los elementos de h deben ser ≥ 0 (con tolerancia numérica)."""
        img = self.aleatoria
        for lam in [0, 0.5, 5]:
            for gamma in [0, 0.5, 5]:
                _, h = funcion_suavizar(img, lam, gamma)
                self.assertTrue(np.all(h >= -1e-8),
                                f"h tiene valores negativos para λ={lam}, γ={gamma}")

    def test_no_modifica_original(self):
        img = self.aleatoria.copy()
        copia = img.copy()
        _, _ = funcion_suavizar(img, 1.0, 1.0)
        np.testing.assert_array_equal(img, copia,
                                      "La función modificó la imagen original (falta .copy())")

def evaluar(func_suavizar, func_dist, func_unif, func_matD, func_ecual):
    global funcion_suavizar, funcion_distribucion, funcion_uniforme, funcion_matrizD, funcion_ecualizar
    funcion_suavizar = func_suavizar
    funcion_distribucion = func_dist
    funcion_uniforme = func_unif
    funcion_matrizD = func_matD
    funcion_ecualizar = func_ecual

    sys.argv = ['']
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuavizarHistograma)
    runner = unittest.TextTestRunner(verbosity=0)
    resultado = runner.run(suite)

    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS: SUAVIZADO DE HISTOGRAMA")
    print("="*70)
    if resultado.wasSuccessful():
        print("¡Todas las pruebas pasaron! La función de suavizado es correcta.")
    else:
        print("Algunas pruebas fallaron. Revisa los mensajes de error arriba.")
        print(f"Fallos: {len(resultado.failures)}")
        print(f"Errores: {len(resultado.errors)}")