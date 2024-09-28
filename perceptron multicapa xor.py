import numpy as np

# Función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptrón Multicapa para el problema XOR
class MLP_XOR:
    def __init__(self, learning_rate=0.1):
        # Inicializar pesos y sesgos (bias) de manera aleatoria
        self.learning_rate = learning_rate
        
        # Pesos entre la capa de entrada (2 neuronas) y la capa oculta (2 neuronas)
        self.weights_input_hidden = np.random.randn(2, 2)
        self.bias_hidden = np.zeros((1, 2))
        
        # Pesos entre la capa oculta (2 neuronas) y la capa de salida (1 neurona)
        self.weights_hidden_output = np.random.randn(2, 1)
        self.bias_output = np.zeros((1, 1))

    def feedforward(self, X):
        # Propagación hacia adelante (cálculo de salidas de la capa oculta y de salida)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backpropagation(self, X, y, output):
        # Error de salida
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Error de la capa oculta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Actualización de los pesos y bias
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # Propagación hacia adelante
            output = self.feedforward(X)

            # Retropropagación y ajuste de pesos
            self.backpropagation(X, y, output)

            # Imprimir pérdida cada 1000 épocas
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Predicción (uso de propagación hacia adelante)
        return self.feedforward(X)

# Datos de entrada y salida para XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas del XOR
y = np.array([[0], [1], [1], [0]])  # Salidas esperadas del XOR

# Inicializar y entrenar el perceptrón multicapa
mlp = MLP_XOR(learning_rate=0.1)
mlp.train(X, y, epochs=10000)

# Hacer predicciones
predicciones = mlp.predict(X)
print(f"Predicciones:\n{predicciones}")
