# Tarea 1 Redes Neuronales
#### Cristóbal Fuentes
###### (Hecho inicialmente en Colaborative, terminado en Spyder porque el editor de Colab es muy malo y no tengo Pycharm)

## Requisitos
Python 3, Numpy, Matplot (use Pip para instalar todo)

## Retrospectiva
Programar una red con Numpy me hace apreciar aún más las pequeñas cosas que librerías comm Pytorch aportan.
Las mayores dificultades que tuve en la tarea fueron el mantener consistente el tamaño y dimensiones de las matrices, en especial porque decidí hacer la red compatible con entradas de datasets desde el principio, lo que llevo a que las derivadas tuviesen dimensiones extras y entonces al momento de multiplicar debía transponer en algunas partes. Tuve también problemas con al función de error, por mucho tiempo mi red no aprendía nada y fue porque estaba aportando el error solo a uno de los outputs (un error tonto pero que me costó un día entero de trabajo jeje...je...fuu). 
Es tremendamente difícil debuggear, ya que incluso si imprimo los pesos en cada iteración, saber como estos aportan al resultado de cada capa requiere hacer calculos manuales, y cuando las capas tienen muchas neurones se vuelve imposibe, así que mucho del debugging es confiar que la red funciona bien y cambiar parámetros hasta que funcione.

## Curvas!
### 1000 epochs
![alt text](https://github.com/solzhen/tarea1nn/blob/master/Figure_1.png "Error")
![alt text](https://github.com/solzhen/tarea1nn/blob/master/Figure_2.png "% Acierto")
![alt text](https://github.com/solzhen/tarea1nn/blob/master/Figure_3.png "% Acierto Zoom")

