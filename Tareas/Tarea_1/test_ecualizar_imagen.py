import unittest
import numpy as np
import sys
from skimage import util, data, exposure

funcion_ecualizar = None
funcion_distribucion = None

class TestEcualizarImagen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Preparar imágenes de prueba"""
        #gradiente lineal
        cls.gradiente = np.linspace(0, 255, 256, dtype=np.uint8).reshape(16, 16)
        # Imagen plana
        cls.plana = np.full((32, 32), 128, dtype=np.uint8)
        #Imagen aleatoria
        np.random.seed(42)
        cls.aleatoria = util.img_as_ubyte(np.random.rand(64, 64))
        #Imagen real de muestra
        cls.coffee = util.img_as_ubyte(data.coffee()[:, :, 0])

    def test_mismo_tamano(self):
        for img in [self.gradiente, self.plana, self.aleatoria, self.coffee]:
            hist = funcion_distribucion(img)
            ecualizada = funcion_ecualizar(img, hist)
            self.assertEqual(ecualizada.shape, img.shape,
                             f"La imagen ecualizada tiene tamaño {ecualizada.shape} en lugar de {img.shape}")

    def test_rango_correcto(self):
        for img in [self.gradiente, self.plana, self.aleatoria, self.coffee]:
            hist = funcion_distribucion(img)
            ecualizada = funcion_ecualizar(img, hist)
            self.assertTrue(np.all(ecualizada >= 0) and np.all(ecualizada <= 255),
                            "La imagen ecualizada contiene valores fuera del rango [0,255]")
            self.assertEqual(ecualizada.dtype, np.uint8,
                             f"El tipo de dato es {ecualizada.dtype}, se espera uint8")

    def test_comparacion_con_skimage(self):
        """La ecualización propia debe dar el mismo resultado que skimage (con tolerancia)."""
        img = self.coffee
        hist = funcion_distribucion(img)
        propia = funcion_ecualizar(img, hist)
        sk = util.img_as_ubyte(exposure.equalize_hist(img))

        self.assertTrue(np.allclose(propia, sk, atol=1),
                        "La ecualización propia difiere de la de scikit-image en más de 1 unidad.")


    def test_imagen_plana(self):
        """Una imagen plana debe permanecer plana (todos iguales) después de ecualizar."""
        img = self.plana
        hist = funcion_distribucion(img)
        ecualizada = funcion_ecualizar(img, hist)
        self.assertTrue(np.all(ecualizada == ecualizada[0,0]),
                        "La imagen plana no permanece plana después de ecualizar.")

    def test_cdf_incluye_nivel(self):
        """Verifica que la CDF incluya el bin correspondiente al nivel."""
        img = np.zeros((10,10), dtype=np.uint8)
        img[:5,:] = 50
        img[5:,:] = 100
        hist = funcion_distribucion(img)
        ecualizada = funcion_ecualizar(img, hist)
        
        self.assertTrue(np.all(ecualizada[img == 50] == ecualizada[0,0]),
                        "Los píxeles con nivel 50 no se mapearon consistentemente.")
        self.assertTrue(np.all(ecualizada[img == 100] == ecualizada[5,0]),
                        "Los píxeles con nivel 100 no se mapearon consistentemente.")
        self.assertTrue(np.all(ecualizada[img == 100] >= 250),
                        "Los píxeles de nivel 100 no se mapearon al blanco (CDF=1).")

    def test_histograma_referencia_externo(self):
        """La ecualización debe usar el histograma proporcionado, no el de la imagen."""
        img = self.coffee
        hist_ref = np.full(256, 1.0/256)
        ecualizada_ref = funcion_ecualizar(img, hist_ref)
        
        hist_propia = funcion_distribucion(img)
        ecualizada_propia = funcion_ecualizar(img, hist_propia)
        
        self.assertFalse(np.allclose(ecualizada_ref, ecualizada_propia, atol=1),
                         "La ecualización con histograma externo dio el mismo resultado que con el propio de la imagen.")

    def test_mayor_contraste(self):
        img = self.coffee
        hist = funcion_distribucion(img)
        ecualizada = funcion_ecualizar(img, hist)
        std_orig = np.std(img)
        std_eq = np.std(ecualizada)
        self.assertGreaterEqual(std_eq, std_orig,
                                "La desviación estándar no aumentó (el contraste no mejoró).")

    
    def test_no_modifica_original(self):
        img = self.gradiente.copy()
        hist = funcion_distribucion(img)
        ecualizada = funcion_ecualizar(img, hist)
        self.assertTrue(np.all(img == self.gradiente),
                        "La función modificó la imagen original. (Consejo: usa img.copy() o genera una matriz nueva)")

def evaluar(funcion_ecualizar_alumno, funcion_distribucion_proporcionada):
    global funcion_ecualizar, funcion_distribucion
    funcion_ecualizar = funcion_ecualizar_alumno
    funcion_distribucion = funcion_distribucion_proporcionada
    
    sys.argv = ['']
    
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEcualizarImagen)
    runner = unittest.TextTestRunner(verbosity=0)
    resultado = runner.run(suite)

    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS: ECUALIZACIÓN DE IMAGEN")
    print("="*70)
    if resultado.wasSuccessful():
        print("¡Todas las pruebas pasaron! La función de ecualización es correcta.")
    else:
        print("Algunas pruebas fallaron. Revisa los mensajes de error arriba.")
        print(f"Fallos: {len(resultado.failures)}")
        print(f"Errores: {len(resultado.errors)}")