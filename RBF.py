import numpy as np

class RBFNN:

    def __init__(self, kernels, centers, beta=1, lr=0.1, epochs=80) -> None:

        self.kernels = kernels # количество используемых RBF, которое также является количеством нейронов в скрытом слое
        self.centers = centers # вектор, содержащий центр каждого RBF
        self.beta = beta       # это гиперпараметр для управления шириной кривой нормального распределения.
        self.lr = lr           # скорость обучения
        self.epochs = epochs   # количество итераций обучения.

        # Случайно инициализированные веса и смещение
        self.W = np.random.randn(kernels, 1)
        self.b = np.random.randn(1, 1)

        # Ошибки
        self.errors = []
        # Градиенты
        self.gradients = []

    # Радикально-базисная функция Гаусса
    def rbf_activation(self, x, center):
        return np.exp(-self.beta * np.linalg.norm(x - center) ** 2)

    # Функция активации
    def linear_activation(self, A):
        return self.W.T.dot(A) + self.b

    # Loss функция
    def least_square_error(self, pred, y):
        return (y - pred).flatten() ** 2

    # Прямое распространение
    def _forward_propagation(self, x):

        # активация RBF (в скрытом слое)
        a1 = np.array([
            [self.rbf_activation(x, center)]
            for center in self.centers
        ])

        # линейная активация (в выходном слое).
        a2 = self.linear_activation(a1)

        return a2, a1

    # Обратное распространение
    def _backpropagation(self, y, pred, a1):
        # Распространение
        dW = -(y - pred).flatten() * a1
        db = -(y - pred).flatten()

        # Обновление весов
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
        return dW, db

    # Тренировочный цикл
    def fit(self, X, Y):

        for _ in range(self.epochs):

            for x, y in list(zip(X, Y)):
                # Прямое распространение
                pred, a1 = self._forward_propagation(x)

                error = self.least_square_error(pred[0], y[0, np.newaxis])
                self.errors.append(error)

                # Обратное распространение
                dW, db = self._backpropagation(y, pred, a1)
                self.gradients.append((dW, db))

    # Функция прогнозирования
    def predict(self, x):
        a2, a1 = self._forward_propagation(x)
        return 1 if np.squeeze(a2) >= 0.5 else 0


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    rbf = RBFNN(kernels=2,
                centers=np.array([
                    [0, 1],
                    [1, 0]

                ]),
                beta=1,
                lr=0.1,
                epochs=50
                )

    rbf.fit(X, Y)


    print(f"RBFN веса : {rbf.W}")
    print(f"RBFN смещение : {rbf.b}")
    print()
    print("-- XOR --")
    print(f"| 1 xor 1 : {rbf.predict(X[3])} |")
    print(f"| 0 xor 0 : {rbf.predict(X[0])} |")
    print(f"| 1 xor 0 : {rbf.predict(X[2])} |")
    print(f"| 0 xor 1 : {rbf.predict(X[1])} |")
    print("_______________")

    print(f"\nОшибки: {rbf.errors}")
    # print(f"Градиенты: {rbf.gradients}")


if __name__ == "__main__":
    main()

