import json
import numpy as np
import matplotlib.pyplot as plt
import time

re = "\033[1;31m"
gr = "\033[1;32m"
cy = "\033[1;36m"
end = "\x1b[0m"

timer = time.strftime("%H:%M")

with open('version.json')as f:
    data = json.load(f)
    version = data.get('version')
    name = data.get('name')
    maintainer = data.get('maintainer')
print(f"                {re + version}v ~ {gr + name} ~ {cy + maintainer}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuroYurii:
    # hidden_size: object

    def __init__(self, input_size, hidden_size, output_size):
        # Ініціалізація розмірів шарів та вагами між ними
        self.hidden_layer_output = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #         Для теста
        self.weights_input_hidden = None

        #         ваги між вхідними і прихованамами шарів `self.weights_input_hidden`
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))

        #          тепер наоборот, ваги між прихованим і вхідним шаром
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        #         Пам'ять для збереження попередніх ваг
        self.memory_weights_input_hidden = None
        self.memory_weights_hidden_output = None

        # Історія помилок,
        # !!!!!!!!!!
        # Зробити потім зберігання помилок у `Json` файл
        self.error_history = []

    def send_signal(self, x):
        #             передаємо сигнал через мережу
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = sigmoid(self.output_layer_input)

        return self.output_layer_output

    def train(self, x, y, epochs=10000, learning_rate=0.1 ): #def 0.1
        for epoch in range(epochs):
            # Передача сегналу уперед:
            output = self.send_signal(x)

            # Обчислення помилки
            error = y - output

            # Збереження помилки в істрію
            total_error = np.mean(np.abs(error))
            self.error_history.append(total_error)

            # Обчислення Градієнт помилки для зворотнього поширення її
            output_delta = error * sigmoid_derivative(output)
            hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

            # Оновлення ваг
            # for i in range(int(6)):
            #     print(i, "Update")
            #     self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
            #     self.weights_input_hidden += np.dot(x.T, hidden_layer_delta) * learning_rate
            #

            self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
            self.weights_input_hidden += np.dot(x.T, hidden_layer_delta) * learning_rate


    def reset_memory(self):
        # Скидання памяті
        self.memory_weights_input_hidden = None
        self.memory_weights_hidden_output = None

    def restore_memory(self):
        # Відновлення попередніх ваг
        if self.memory_weights_input_hidden is not None and self.memory_weights_hidden_output is not None:
            self.memory_weights_input_hidden = self.memory_weights_input_hidden.copy()
            self.weights_hidden_output = self.memory_weights_hidden_output.copy()

    def __str__(self):
        return f"NeuroYurii(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

# Приклад на навчанні бінарних класифікаціях
x = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]

    ]
)

y = np.array(
    [
        [0],
        [1],
        [0],
        [1]
    ]
)


input_size = 2
hidden_size = 4
output_size = 1


print("Starting")
for i in range(50):

    # Сама нейронка
    nn = NeuroYurii(input_size, hidden_size, output_size)
    # Навчання
    nn.train(x, y)
    print(f"{i}. {nn}")

    import time
    current_time = time.strftime("%H:%M:%S")

    # Будуємо графіки
    plt.plot(range(len(nn.error_history)), nn.error_history)
    plt.title(f'Training Progress {version}v\n{current_time}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute ')
    plt.savefig(f'plots/{timer}training_plot.png')