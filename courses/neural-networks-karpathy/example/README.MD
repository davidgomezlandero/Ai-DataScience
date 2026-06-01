## Activation softmax
Consiste en la salida de la ultima capa intermedia de la red neuronal que nos da el resultado la convertimos para que los resultados nos den entre 0 y 1 dando una suma de 1 en total y conocer cual tiene mayor probabilidad de ser la respuesta sobre 1.

Lo que se hace es calcular su exp(input) y luego se normaliza con exp(input) / sum(todos los exp(input)).

## Loss with categorical cross-entropy

Lsubi = - sum(y(subi,j) log(ŷ(subi,j)))
	Lsubi - sample loss value
	y - target values
	ŷ - predicted values
	Lsubi - sample loss value
	i - i-th sample in a set
	j - label/output index

Simplificacion debido a la codificación one-hot
	Lsubi = -log(ŷ(subi,k))
		k - target label index, index of correct class probability

One-hot encoding
	Vector con n clases de longitud donde la clase objetivo tendra 1. 
		Classes: 2
		Label:  0
		One-hot: [1, 0]