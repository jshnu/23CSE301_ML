import matplotlib.pyplot as plt


def summation_unit(inputs, weights, bias):
    total = bias

    for i in range(len(inputs)):
        total += inputs[i] * weights[i]

    return total


def step_activation(value):
    if value >= 0:
        return 1
    else:
        return 0


def train_perceptron(inputs, targets, initial_weights, initial_bias,
                     learning_rate, max_epochs=1000,
                     convergence_error=0.002):

    weights = initial_weights.copy()
    bias = initial_bias

    for epoch in range(1, max_epochs + 1):

        for i in range(len(inputs)):

            net_input = summation_unit(
                inputs[i],
                weights,
                bias
            )

            predicted = step_activation(net_input)

            error = targets[i] - predicted

            bias = bias + learning_rate * error

            for j in range(len(weights)):
                weights[j] = (
                    weights[j]
                    + learning_rate * error * inputs[i][j]
                )

        
        sum_squared_error = 0

        for i in range(len(inputs)):

            net_input = summation_unit(
                inputs[i],
                weights,
                bias
            )

            predicted = step_activation(net_input)

            error = targets[i] - predicted

            sum_squared_error += error ** 2

        if sum_squared_error <= convergence_error:
            return epoch

    return max_epochs




inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

targets = [0, 0, 0, 1]

initial_weights = [0.2, -0.75]
initial_bias = 10

learning_rates = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
]

iterations = []

for learning_rate in learning_rates:

    epochs = train_perceptron(
        inputs,
        targets,
        initial_weights,
        initial_bias,
        learning_rate
    )

    iterations.append(epochs)

print("Learning Rate    Iterations")

for i in range(len(learning_rates)):
    print(
        learning_rates[i],
        "             ",
        iterations[i]
    )



plt.plot(learning_rates, iterations, marker="o")

plt.xlabel("Learning Rate")
plt.ylabel("Iterations to Converge")
plt.title("Learning Rate vs Convergence Iterations")

plt.grid(True)
plt.show()
