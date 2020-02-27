# Introduccion

Para el concurso ZaraTech Challenge 2020 se nos dió el problema: A partir de la foto de una prenda encontrar prendas similares.

# Solución

La solución que hemos encontrado para este problema es:

1. Una aplicación web donde puedas subir las fotos.
1. Un servidor que reciba la foto.
1. El servidor tendrá que procesar la foto (Convertirla a los formatos necesarios).
1. Un modelo (Modelo 1) tendrá que identificar que tipo de prenda es.
1. Posteriormente se pasará la foto al siguiente modelo para que este la codifique.
1. Ahora tocara calcular la similitud con la matriz de prendas que tenemos ya calculadas del entrenamiento.
1. Una vez tenemos las similitudes con las prendas las ordenamos de mayor a menor.
1. Mandamos la lista al usuario de vuelta.

Para esta solución hemos escogido crear un modelo que directamente clasifique los distintos tipos de prenda. Porbablemente, como no tenemos muchos datos ni mucha capacidad de entrenamiento para los modelos lo mejor sería crear 1 modelo para cada tipo de ropa y que cada uno pueda dar su versión de la imagen y coger el tipo en función de la seguridad que tienen los modelos de que esa imagen sea de su tipo o no. 

Sin, embargo no tenemos la capacidad de entrenar tantos modelos ni de ejecutarlos en paralelo a la hora del desarrollo por lo que solo vamos a entrenar un modelo, el que va a seleccionar los tipos de los modelos.

# Implementación 

A la hora de implementar la solución que hemos presentado más arriba nos hemos encontrado con bastantes problemas:

1. No teniamos ningún dataset para entrenar los modelos. esto ha complicado bastante la tarea puesto que no sabemos cuales pueden ser los criterios para decir que la ropa es similar.
1. Los modelos que utilizan imagenes se tienen que entrenar preferiblemente en GPU por lo que una vez hemos pasado a la fase de entrenamiento. Entrenar los modelos es bastante costoso. 
1. Las imagenes del dataset que hemos conseguido a veces contienen más elementos u otras prendas lo que va a dificultar que el modelo aprenda rápido.

Vamos a utilizar una ResNet como modelo para determinar el tipo de la imagen. Para entrenar el modelo usaremos un dataset: [DataFashion 2](https://github.com/switchablenorms/DeepFashion2) para crear los tipos de imagenes del modelo usaremos las anotaciones de las imagenes. Sacaremos los items que hay en ella y clasificaremos esa imagen dentro de esos grupos.

# Requisitos

Para poder ejecutar en local tenemos que tener instalados los siguientes paquetes:

Tensorflow

```
pip intsall tensorflow
```
Django
```
pip intsall django
```
Matplotlib
```
pip intsall matplotlib
```
Pillow
```
pip intsall pillow
```

# Estructura del Repositorio

```
zara-tech
├── training: Directorio con los scripts de entrenamiento.
│   └── type_model.py: Fichero para preparar el dataset y entrenar el modelo.
│
└── zara: Servidor Django
│   └── zara: Lógica del servidor
│   |   ├── photo.py: Lógica para implementar las acciones con las fotos.
│   |   ├── views.py: Lógica del servidor.
│   │   │   
│   |   └── models: Directorio con los recursos para operar con los modelos.
 
```