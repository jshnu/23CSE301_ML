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


def calculate_error(target, predicted):
    return target - predicted


def train_perceptron(inputs, targets, weights, bias, learning_rate,
                     max_epochs=1000, convergence_error=0.002):

    epoch_errors = []

    for epoch in range(1, max_epochs + 1):

        for i in range(len(inputs)):

            net_input = summation_unit(
                inputs[i],
                weights,
                bias
            )

            predicted = step_activation(net_input)

            error = calculate_error(targets[i], predicted)

            
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

        epoch_errors.append(sum_squared_error)

        if sum_squared_error <= convergence_error:
            return weights, bias, epoch, epoch_errors

    return weights, bias, max_epochs, epoch_errors




inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

targets = [0, 0, 0, 1]

weights = [0.2, -0.75]
bias = 10

learning_rate = 0.05

weights, bias, epochs, errors = train_perceptron(
    inputs,
    targets,
    weights,
    bias,
    learning_rate
)

print("Final weights:")
print("W0 =", bias)
print("W1 =", weights[0])
print("W2 =", weights[1])

print("\nNumber of epochs:", epochs)
print("Final SSE:", errors[-1])

print("\nFinal predictions:")

for i in range(len(inputs)):

    net_input = summation_unit(
        inputs[i],
        weights,
        bias
    )

    prediction = step_activation(net_input)

    print(
        inputs[i],
        "->",
        prediction,
        "(Target:",
        targets[i],
        ")"
    )


epoch_numbers = range(1, len(errors) + 1)

plt.plot(epoch_numbers, errors)

plt.xlabel("Epoch")
plt.ylabel("Sum-Square Error")
plt.title("AND Gate Perceptron: Epoch vs Error")

plt.grid(True)
plt.show()
