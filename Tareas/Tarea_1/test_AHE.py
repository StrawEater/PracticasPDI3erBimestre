import unittest
import numpy as np
import sys
from skimage import util, data

funcion_ajustable = None
funcion_distribucion = None
funcion_uniforme = None
funcion_ecualizar = None

class TestEcualizacionAjustable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Preparar imágenes de prueba (sintéticas)"""
        cls.gradiente = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
        cls.plana = np.full((32, 32), 128, dtype=np.uint8)
        np.random.seed(42)
        cls.aleatoria = util.img_as_ubyte(np.random.rand(64, 64))
        cls.coffee = util.img_as_ubyte(data.coffee()[:, :, 0])

    def test_mismo_tamano(self):
        """La imagen de salida debe tener el mismo tamaño que la entrada."""
        for img in [self.gradiente, self.plana, self.aleatoria, self.coffee]:
            for lam in [0, 1, 10]:
                ecualizada = funcion_ajustable(img, lam)
                self.assertEqual(ecualizada.shape, img.shape,
                                 f"Tamaño incorrecto para λ={lam}: {ecualizada.shape} vs {img.shape}")

    def test_rango_correcto(self):
        """La imagen ecualizada debe estar en [0,255] y ser uint8."""
        for img in [self.gradiente, self.plana, self.aleatoria, self.coffee]:
            for lam in [0, 0.5, 5, 100]:
                ecualizada = funcion_ajustable(img, lam)
                self.assertTrue(np.all(ecualizada >= 0) and np.all(ecualizada <= 255),
                                f"Valores fuera de rango para λ={lam}")
                self.assertEqual(ecualizada.dtype, np.uint8,
                                 f"Tipo incorrecto para λ={lam}: {ecualizada.dtype}")

    def test_lambda_cero_igual_standard(self):
        """Con λ=0, debe dar el mismo resultado que ecualizar_imagen con el histograma propio."""
        img = self.coffee
        hi = funcion_distribucion(img)
        std_eq = funcion_ecualizar(img, hi)
        ajustable_eq = funcion_ajustable(img, 0.0)   # λ=0
        np.testing.assert_array_equal(std_eq, ajustable_eq,
                                      "λ=0 no coincide con la ecualización estándar.")


    def test_no_modifica_original(self):
        """La imagen original no debe ser modificada."""
        img = self.coffee.copy()
        copia = img.copy()
        _ = funcion_ajustable(img, 1.0)
        np.testing.assert_array_equal(img, copia,
                                      "La función modificó la imagen original (falta .copy() o se hace in-place).")

    def test_contraste_aumenta_con_cero_luego_disminuye(self):
        """El contraste (std) debe aumentar con λ=0 respecto al original y luego disminuir con λ grande."""
        img = self.coffee
        std_orig = np.std(img)
        eq0 = funcion_ajustable(img, 0.0)
        eq1 = funcion_ajustable(img, 1.0)
        eq10 = funcion_ajustable(img, 10.0)
        self.assertGreater(np.std(eq0), std_orig,
                           "λ=0 no aumentó el contraste respecto al original.")
        self.assertLess(np.std(eq10), np.std(eq1),
                        "λ=10 debería tener menor contraste que λ=1.")

    def test_imagen_plana(self):
        """Una imagen plana debe permanecer plana (todos iguales) para cualquier λ."""
        img = self.plana
        for lam in [0, 1, 10, 100]:
            eq = funcion_ajustable(img, lam)
            self.assertTrue(np.all(eq == eq[0,0]),
                            f"Imagen plana con λ={lam} no resultó plana.")

def evaluar(funcion_ajustable_alumno, funcion_distribucion_alumno,
            funcion_uniforme_alumno, funcion_ecualizar_alumno):

    global funcion_ajustable, funcion_distribucion, funcion_uniforme, funcion_ecualizar
    funcion_ajustable = funcion_ajustable_alumno
    funcion_distribucion = funcion_distribucion_alumno
    funcion_uniforme = funcion_uniforme_alumno
    funcion_ecualizar = funcion_ecualizar_alumno

    sys.argv = ['']
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEcualizacionAjustable)
    runner = unittest.TextTestRunner(verbosity=0)
    resultado = runner.run(suite)

    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS: ECUALIZACIÓN AJUSTABLE")
    print("="*70)
    if resultado.wasSuccessful():
        print("¡Todas las pruebas pasaron! La función ajustable es correcta.")
    else:
        print("Algunas pruebas fallaron. Revisa los mensajes de error arriba.")
        print(f"Fallos: {len(resultado.failures)}")
        print(f"Errores: {len(resultado.errors)}")