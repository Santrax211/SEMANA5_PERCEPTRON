import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Inicialización de pesos y bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entrenamiento del modelo
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Actualización de pesos y bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

# Datos de ejemplo
X = np.array([[1, 1], [2, 2], [1.5, 1.5], [3, 3]])
y = np.array([0, 0, 1, 1])  # Etiquetas binarias

# Inicializar y entrenar el perceptrón
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X, y)

# Predicción
predicciones = perceptron.predict(X)
print(f"Predicciones: {predicciones}")
