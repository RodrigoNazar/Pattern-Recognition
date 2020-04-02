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

Finalmente ejecutar:

```
python main.py
```

Para ejecutar la tarea.

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Autor

* **Rodrigo Nazar** - *Ingeniería Eléctrica UC* - [RodrigoNazar](https://github.com/RodrigoNazar)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


## Comandos útiles

* Revisar https://www.youtube.com/watch?v=CdltAssTMs8&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=15
