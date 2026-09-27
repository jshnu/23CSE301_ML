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



inputs = [1, 1]
weights = [0.2, -0.75]
bias = 10

target = 1

summation = summation_unit(inputs, weights, bias)
output = step_activation(summation)
error = calculate_error(target, output)

print("Summation output:", summation)
print("Step activation output:", output)
print("Error:", error)
