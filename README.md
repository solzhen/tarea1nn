# Tarea 1 Redes Neuronales
#### Cristóbal Fuentes
###### (Hecho inicialmente en Colaborative, terminado en Spyder porque el editor de Colab es muy malo y no tengo Pycharm)

## Requisitos
Python 3, Numpy, Matplot (use Pip para instalar todo)

## Uso
Corra tarea.py.
Modifique tarea.py para cambiar funciones de activación, entradas, etc. El código está hecho para funcionar con Iris DB, pero cambiando unos pocos valores se puede hacer genérico. No lo hice porque me faltó tiempo (¿qué onda los profes del DCC dejando las tareas para la misma semana? entiendo que la siguiente no se puede evaluar pero whyyy)

## Retrospectiva
Programar una red con Numpy me hace apreciar aún más las pequeñas cosas que librerías como Pytorch aportan.

Las mayores dificultades que tuve en la tarea fueron el mantener consistente el tamaño y dimensiones de las matrices, en especial porque decidí hacer la red compatible con entradas de datasets desde el principio, lo que llevo a que las derivadas tuviesen dimensiones extras y entonces al momento de multiplicar debía transponer en algunas partes. Tuve también problemas con al función de error, por mucho tiempo mi red no aprendía nada y fue porque estaba aportando el error solo a uno de los outputs (un error tonto pero que me costó un día entero de trabajo jeje...je...fuu). 

Es tremendamente difícil debuggear, ya que incluso si imprimo los pesos en cada iteración, saber como estos aportan al resultado de cada capa requiere hacer calculos manuales, y cuando las capas tienen muchas neuronas se vuelve imposibe, así que mucho del debugging es confiar que la red funciona bien y cambiar parámetros hasta que funcione.

Inicialmente había hecho la tarea con neuronas y listas de neuronas, pero era muy lento. Como ventaja cada neurona podía tener su propia memoria facilitanto el backpropagation y el parameter updating, pero era tremendamente lento. Así que decidí usar numpy arrays.

Como se ven en las curvas de abajo, con 10000 epochs la red llega a 95% de precisión. En mi computador se demora cerca de 1 minuto, pero depende (obviamente) de la red. En los graficos de abajo se entrenó sobre 5 capas de (15,10,12,8,3) neuronas respectivamente con Tanh como función de activación en todas ellas.
## Curvas!
Las siguientes curvas con Tanh en cada capa
### 1000 epochs 
#### Error
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_1.png "Error")
#### % Acierto
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_2.png "% Acierto")
#### % Acierto (acercamiento en últimos epochs)
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_3.png "% Acierto Zoom")

### 10000 epochs
#### Error
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_4.png "Error")
#### % Error (acercamiento en últimos epochs)
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_5.png "% Acierto Zoom")
#### % Acierto
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_6.png "% Acierto")
#### % Acierto (acercamiento en últimos epochs)
![alt text](https://github.com/solzhen/tarea1nn/blob/master/figures/Figure_7.png "% Acierto Zoom")
