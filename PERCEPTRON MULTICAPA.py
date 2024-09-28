import numpy as np

# Función de activación (sigmoide) y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptrón Multicapa
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inicialización de los pesos
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Pesos entre capa de entrada y oculta
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        # Pesos entre capa oculta y de salida
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def feedforward(self, X):
        # Propagación hacia adelante
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backpropagation(self, X, y, output):
        # Calcula el error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Actualización de los pesos y los bias
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Paso de feedforward
            output = self.feedforward(X)

            # Paso de retropropagación
            self.backpropagation(X, y, output)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.feedforward(X)

# Datos de ejemplo (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicializamos la red con 2 neuronas de entrada, 2 ocultas y 1 de salida
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

# Entrenamos el modelo
mlp.train(X, y, epochs=10000)

# Probamos el modelo
predicciones = mlp.predict(X)
print(f"Predicciones: {predicciones}")
