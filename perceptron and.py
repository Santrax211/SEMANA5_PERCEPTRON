import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Inicializar pesos y bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entrenamiento del perceptrón
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Actualizar pesos y bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        # Función de activación (Escalón)
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

# Datos de entrada y salida para la compuerta AND
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])  # Salidas de la compuerta AND

# Crear el perceptrón y entrenarlo
perceptron = Perceptron(learning_rate=0.1, n_iter=10)
perceptron.fit(X, y)

# Probar el modelo
predictions = [perceptron.predict(x) for x in X]

# Resultados
print("Entradas: \n", X)
print("Predicciones: \n", predictions)
