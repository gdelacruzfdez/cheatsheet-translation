**1. Deep Learning cheatsheet**

&#10230;Hoja de referencia de Aprendizaje Profundo.

<br>

**2. Neural Networks**

&#10230; Redes Neuronales

<br>

**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230;Las Redes Neuronales (en inglés _neural networks_) son una clase de modelos contruidos a base de capas de neurones. Entre los tipos de redes neuronales más habitualmente utilizados se incluyen las redes neuronales convolucionales y las recurrentes.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230;Arquitectura ― El vocabulario relacionado con las arquitecturas de las redes neuronales se describe en la figura siguiente.

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230;[Capa de entrada, capa oculta, capa de salida]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230;Siendo i la i-ésima capa de la red y j la j-ésima unidad oculta de la capa, tenemos:

<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230;donde w, b y z representan el peso, el sesgo y la salida respectivamente.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230;Función de activación ― Las funciones de activación se utilizan al final de una unidad oculta para introducir complejidades no lineales al modelo. Aquí se presentan las más comunes:

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;[Sigmoide, Tanh,ReLU, Leaky ReLU]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230;Pérdida de entropía cruzada ― En el contexto de las redes neuronales, la pérdida de entropía cruzada L(z,y) es comúnmente utilizada y definida de la siguiente manera:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230;Tasa de aprendizaje ― La tasa de aprendizaje (en inglés, _learning rate_), a menudo definida como α o en ocasiones como η, indica a que ritmo se actualizan los pesos. Esto puede ser fijo o cambiado de forma adaptativa. Actualmente el método más popular se llama Adam, el cual es un método que adapta la tasa de aprendizaje.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230;Propagación hacia atrás - La propagación hacia atrás (en inglés, _backpropagation_) es un método para actualizar los pesos de la red neuronal teniendo en cuenta la salida actual y la salida deseada. La derivada con respecto al peso w se calcula utilizando la regla de la cadena y se escribe de la siguiente manera:

<br>

**13. As a result, the weight is updated as follows:**

&#10230;Por consiguiente, el peso se actualiza de la siguiente manera:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230;Actualizar los pesos ― En una red neuronal, los pesos se actualizan de la siguiente manera:

<br>

**15. Step 1: Take a batch of training data.**

&#10230;Paso 1: Coge un lote de datos de entrenamiento.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230;Paso 2: Realiza la propagación hacia delante para obtener la pérdida correspondiente.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230;Paso 3: Realizar una propagación hacia atrás de la perdida para obtener los gradientes.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230;Paso 4: Usar los gradientes para actualizar los pesos de la red.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230;Dropout ― El dropout es una tecnica utilizada para prevenir el sobreajuste al conjunto de entrenamiento abandonando unidades en una red neuronal. En la práctica, los neurones son abandonados con una probabilidad p o conservados con una probabilidad 1-p

<br>

**20. Convolutional Neural Networks**

&#10230;Redes Neuronales Convolucionales

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230;Requisito de la capa convolucional ― Siendo W el tamaño del volumen de entrada, F el tamaño de los neurones de la capa convolucional y P la cantidad de zero padding, el numero de neurones N que caben en un volumen dado es tal que:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230;Batch normalization ― Se trata de un paso de los hiperparametros γ,β que normaliza el batch {xi}. Siendo μB,σ2B la media y la varianza de lo que queremos corregir al batch, se hace de la siguiente manera: 

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230;Habitualmente se realiza despues de una capa totalmente conectada/convolucional y antes de una capa no lineal y tiene como objetivo permitir mayores tasas de aprendizaje y reducir la fuerte dependencia de la inicialización.

<br>

**24. Recurrent Neural Networks**

&#10230;Redes Neuronales Recurrentes

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230;Tipos de puertas ― Aquí están los diferentes tipos de puertas que encontramos en una red neuronal recurrente típica.

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230;[Puerta de entrada, puerda del olvido, puerta, puerta de salida]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230;[¿Escribir en la célula o no?, ¿Borrar una célula o no?, ¿Cuánto escribir en la célula?, ¿En qué punto revelar la célula?]

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230;LSTM ― Una red de larga memoria a corto plazo (en inglés _long short-term memory network_, LSTM) es un tipo de modelo RNN que evita el problema de _vanishing gradient_ añadiendo puertas del olvido. 

<br>

**29. Reinforcement Learning and Control**

&#10230;Aprendizaje por Refuerzo y Control.

<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230;El objetivo del aprendizaje por refuerzo es, para un agente, aprender a como evolucionar en un entorno.

<br>

**31. Definitions**

&#10230;Definiciones

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230;Procesos de decision de Markov ― Un proceso de decisión de Markov es una 5-tupla (S,A,{Psa},γ,R) donde:

<br>

**33. S is the set of states**

&#10230;S es el conjunto de estados

<br>

**34. A is the set of actions**

&#10230;A es el conjunto de acciones

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230;{Psa} son las probabilidades de transición de estado para s∈S y a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230;γ∈[0,1[ es el factor de descuento

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230;R:S×A⟶R or R:S⟶R es la función de recompensa que el algoritmo quiere maximizar.

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;Política ― Una política π es una función π:S⟶A que mapea estados a acciones.

<br>

**39. Remark: we say that we execute a given policy π if given a state s we take the action a=π(s).**

&#10230;Nota: Decimos que podemos ejecutar una política π dada si dado un estado s tomamos la acción a=π(s).

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;Función de valor ― Para una política π dada y un estado s dado, definimos la función de valor Vπ de la siguiente manera:  

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;Ecuación de Bellman ― Las ecuaciones óptimas de Bellman caracterizan la función de valor Vπ∗ de la política óptima π∗:   

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;Nota: Decimos que la política óptima π∗ para un estado s dado es tal que:

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;Algoritmo de iteración de valor ― El algoritmo de iteración de valor consta de dos pasos: 

<br>

**44. 1) We initialize the value:**

&#10230;1) Inicializamos el valor:

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;2) Iteramos el valor basandonos en los valores anteriores:

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;Estimación de la maxima verosimilitud ― Las estimaciones de la máxima verosimilitud para las probabilidades de transición de estados son de la siguiente manera: 

<br>

**47. times took action a in state s and got to s′**

&#10230; veces que se tomó la acción a en el estado s y se obtuvo s'

<br>

**48. times took action a in state s**

&#10230; veces que se tomó la acción a en el estado s

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;Q-learning ―  Q-learning es una estimación no paramétrica de Q, que se realiza de la siguiente manera:

<br>

**50. View PDF version on GitHub**

&#10230; Ver la versión PDF en GitHub

<br>

**51. [Neural Networks, Architecture, Activation function, Backpropagation, Dropout]**

&#10230;[Redes Neuronales, Arquitectura, Función de Activación, Propagación hacia atrás, Dropout]

<br>

**52. [Convolutional Neural Networks, Convolutional layer, Batch normalization]**

&#10230; [Redes Neuronales Convolucionales, Capa Convolucional, Batch normalization]

<br>

**53. [Recurrent Neural Networks, Gates, LSTM]**

&#10230; [Redes Neuronales Recurrentes, Puertas, LSTM]

<br>

**54. [Reinforcement learning, Markov decision processes, Value/policy iteration, Approximate dynamic programming, Policy search]**

&#10230; [Aprendizaje por refuerzo, Procesos de decisión de Markov, Iteración de Valor/Política, Programación dinámica aproximada, Busqueda de políticas]
