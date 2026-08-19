import unittest
import numpy as np
import sys

funcion_a_evaluar = None

class TestCalcularProbabilidadAcumulada(unittest.TestCase):

    def test_nivel_cero(self):
        """Si nivelGris = 0, la CDF debe ser la probabilidad del bin 0."""
        distribucion = np.zeros(256)
        distribucion[0] = 0.3 # 1er bin es 30%
        distribucion[1] = 0.7 # 2do bin es 70%
        resultado = funcion_a_evaluar(distribucion, 0)
        esperado = 0.3
        if not np.isclose(resultado, esperado):
            mensaje = (f"ERROR: Para nivelGris=0, la CDF debe ser {esperado}, "
                       f"pero se obtuvo {resultado}. "
                       "Posible error: estás usando range(nivelGris) en lugar de range(nivelGris+1), "
                       "o no estás incluyendo el bin 0 en la suma.")
            self.fail(mensaje)

    def test_nivel_ultimo(self):
        """Para nivelGris = 255, la CDF debe ser 1.0"""
        distribucion = np.full(256, 1.0/256) # Distribucion uniforme
        resultado = funcion_a_evaluar(distribucion, 255)
        esperado = 1.0
        if not np.isclose(resultado, esperado):
            mensaje = (f"ERROR: Para nivelGris=255, la CDF debe ser {esperado}, "
                       f"pero se obtuvo {resultado}. "
                       "Posible error: no estás sumando todos los elementos, "
                       "o la distribución no está normalizada.")
            self.fail(mensaje)

    def test_cdf_uniforme_lineal(self):
        """Para distribución uniforme, CDF(k) = (k+1)/256."""
        distribucion = np.full(256, 1.0/256)
        for k in [0, 10, 50, 127, 200, 254]:
            esperado = (k + 1) / 256
            resultado = funcion_a_evaluar(distribucion, k)
            if not np.isclose(resultado, esperado):
                mensaje = (f"ERROR: Para k={k}, CDF uniforme esperada={esperado}, "
                           f"pero se obtuvo {resultado}. "
                           "Posible error: el índice o el rango de suma no es correcto.")
                self.fail(mensaje)

    def test_distribucion_con_pico(self):
        """Distribución con toda la masa en el nivel 128."""
        distribucion = np.zeros(256)
        distribucion[128] = 1.0
        #Para niveles < 128, CDF = 0
        for k in [0, 10, 127]:
            resultado = funcion_a_evaluar(distribucion, k)
            esperado = 0.0
            if resultado != esperado:
                mensaje = (f"ERROR: Para k={k}, CDF debería ser {esperado}, "
                           f"pero se obtuvo {resultado}. "
                           "Posible error: estás sumando mal los índices.")
                self.fail(mensaje)
        #Para niveles >= 128, CDF = 1
        for k in [128, 200, 255]:
            resultado = funcion_a_evaluar(distribucion, k)
            esperado = 1.0
            if not np.isclose(resultado, esperado):
                mensaje = (f"ERROR: Para k={k}, CDF debería ser {esperado}, "
                           f"pero se obtuvo {resultado}. "
                           "Posible error: no estás sumando hasta el nivel correcto.")
                self.fail(mensaje)

    def test_distribucion_no_normalizada(self):
        """La función debe sumar los valores tal cual."""
        distrib = np.zeros(256)
        distrib[:5] = [1, 2, 3, 4, 5]  #suma = 15
        #Para nivel 2: 1+2+3 = 6
        resultado = funcion_a_evaluar(distrib, 2)
        esperado = 6
        if resultado != esperado:
            mensaje = (f"ERROR: Para distrib no normalizada en k=2, esperado={esperado}, "
                       f"obtenido={resultado}. "
                       "Posible error: estás normalizando internamente, lo que no debe hacerse.")
            self.fail(mensaje)
        #Para nivel 4: 15
        resultado = funcion_a_evaluar(distrib, 4)
        esperado = 15
        if resultado != esperado:
            mensaje = (f"ERROR: Para distrib no normalizada en k=4, esperado={esperado}, "
                       f"obtenido={resultado}. "
                       "Posible error: no estás sumando todos los elementos.")
            self.fail(mensaje)

    def test_manejo_indices_fuera_rango(self):
        """Si nivelGris < 0 o > 255, debe manejarse sin error."""
        distrib = np.full(256, 1.0/256)
        #vemos con -1
        try:
            resultado = funcion_a_evaluar(distrib, -1)
            esperado = 0.0
            if not np.isclose(resultado, esperado):
                mensaje = (f"ERROR: Para nivelGris=-1, esperado={esperado}, obtenido={resultado}. "
                           "Posible error: no estás acotando los índices negativos.")
                self.fail(mensaje)
        except Exception as e:
            self.fail(f"ERROR: nivelGris=-1 lanzó excepción: {e}. "
                      "Debes acotar los índices antes de sumar.")

        #vemos con 300
        try:
            resultado = funcion_a_evaluar(distrib, 300)
            esperado = 1.0
            if not np.isclose(resultado, esperado):
                mensaje = (f"ERROR: Para nivelGris=300, esperado={esperado}, obtenido={resultado}. "
                           "Posible error: no estás acotando los índices >255.")
                self.fail(mensaje)
        except Exception as e:
            self.fail(f"ERROR: nivelGris=300 lanzó excepción: {e}. "
                      "Debes acotar los índices antes de sumar.")


def evaluar(funcion_alumno):
    """
    Esta es la función que el alumno llamará desde Colab pasándole su función.
    """
    global funcion_a_evaluar
    funcion_a_evaluar = funcion_alumno
    
    sys.argv = ['']
    
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCalcularProbabilidadAcumulada)
    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)

    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    if resultado.wasSuccessful():
        print("¡Todas las pruebas pasaron! La función es correcta.")
    else:
        print("Algunas pruebas fallaron. Revisa los mensajes de error arriba.")
        print(f"Fallos: {len(resultado.failures)}")
        print(f"Errores: {len(resultado.errors)}")