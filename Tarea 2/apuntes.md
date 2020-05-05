# Apuntes

## LBP

Características de textura, que consiste en 3 pasos:

  * Coding: En cada píxel codificar la información de la comparación de el con los píxeles vecinos.

  * Mapping: Mapea cada uno de los valores a un cierto patrón. Son 59 patrones distintos.

  * Histogram: Armar un histograma de cuantas veces aparece cada patrón en el mapeo. Así se obtiene un vector de 59 características.

Imágenes similares tienen LBP's similares.

## Texturas de Haralick

Características en las que se elige un vector de dirección, se calcula la matriz de coocurrencia y a partir de ella, se obtienen distintas características como:

  * Contraste
  * Momento de diferencia inversa
  * Energía
  * Entropía

## Gabor

Se utilizan una familia de funciones base para luego convolucionarla con la imagen original. Luego se suman los elementos de la matriz y se obtiene una característica.

El número de características es el número de funciones de Gabor que se utilizan.
