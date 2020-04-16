# T1-Patrones

Esta tarea consiste en desarrollar un reconocedor atomático de las letras A, S, D, F y G de distintas fuentes.

## Empezando

Estas instrucciones le proporcionarán las instrucciones para poder ejecutar de manera correcta en su máquina local la tarea.

### Prerrequisitos

Las cosas más importantes a tener en consideración, son:

* La tarea fue implementada usando un ambiente virtual en Python3.6, por lo que se recomienda fuertemente hacer lo mismo.

  A continuación proveeré unos links para su instalación y uso:

  [Instalación y uso en Ubuntu](https://www.digitalocean.com/community/tutorials/como-instalar-python-3-y-configurar-un-entorno-de-programacion-en-ubuntu-18-04-guia-de-inicio-rapido-es)

  [Instalación y uso en Mac](https://sourabhbajaj.com/mac-setup/Python/virtualenv.html)

  [Instalación y uso en Windows](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/)

  Luego de tener el ambiente virtual, es necesario ingresar a el para los pasos que vienen.

* Poder ejecutar el comando ```make``` dentro del directorio, al igual que ```make clean```.

### Instalación

Lo primero es instalar el ambiente virtual de python 3 en el directorio con:

```
virtualenv -p python3 venv
```

Para luego ingresar a el con:

```
source venv/bin/activate
```

Una vez ya dentro del ambiente virtual se recomienda ejecutar los siguientes comandos:

Instalar las librerías utilizadas en la tarea vía pip:
```
pip install -r requirements.txt
```

Luego ejecutar:

```
make clean
make
```

Esto hace que se eliminen los archivos compilados de python ```.pyc``` y que se ejecute el arhivo ```setup.py``` que ordena las dependencias de las rutas del proyecto.

Si no puedes ejecutar el comando ```make```, puedes ejecutar el siguiente comando que lo reemplaza:
```
python setup.py develop
```


Finalmente ejecutar:

```
python main.py
```

Para ejecutar el código de main de la tarea.

## Corriendo e importando el reconocedor

En el enunciado de la tarea se pide implementar una función ```reconocedor.py``` con una función ```reconocedor``` dentro.

Para importar esta función en un módulo aparte (como me imagino que corregirán), es necesario hacer todos los pasos anteriores, para que las rutas entre módulos estén bien definidas y no haya errores del tipo "Cannot import module..."

Entonces se debe copiar el archivo ```.py``` con el que se testeará la función ```reconocedor.py``` en el directorio ```T1-Patrones``` y desde ese módulo importar el reconocedor como:

```
from reconocedor import reconocedor
```

Así, se podrá usar la función de correcta manera

### Módulos y Directorios

* ```apps```: Se encuentran las distintas aplicaciones desarrolladas para cada uno de los requerimientos de la tarea.

  * ```feature_setup.py```: Procesa las imágenes de la ruta ```img``` y obtiene sus características (tanto como de training, como testing), para guardarlas en ```data```.

  * ```testing.py```: Ejecuta el clasificador de cada una de las imágenes de testing sobre las de training, consultando la información guardada en ```data```. También contiene el test de combinaciones de distintos momentos de hu.

  * ```statistics.py```: Módulo que muestra estadísticas de desempeño del algoritmo utilizado.

* ```data```: Directorio en el que se guarda la información de las características obtenidas de las imágenes de testing y training.

* ```img```: Directorio en el que se encuentran las imágenes.

* ```results```: Directorio en el que se guarda un archivo con los resultados obtenidos del test ```huCombinationTest``` ubicado en ```apps/testing.py```. Lo que hace este test es iterar entre distintas combinaciones de momentos de hu, para elegir la que tiene mayor porcentaje de aciertos.

* ```utils```: Directorio de útiles desarrollados para la tarea.

  * ```huMoments.py```: Está toda la implementación de los momentos de hu.

  * ```utils.py```: útiles generales para el procesamiento de imágenes


## Autor

* **Rodrigo Nazar** - *Ingeniería Eléctrica UC* - [RodrigoNazar](https://github.com/RodrigoNazar)
