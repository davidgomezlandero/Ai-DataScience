import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.loss(softmax_output[0]) * target_output[0] +
         math.loss(softmax_output[1]) * target_output[1] +
         math.loss(softmax_output[2]) * target_output[2])

print(loss)
