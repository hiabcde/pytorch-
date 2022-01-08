import numpy
import random
import torch

from model import *

x = [random.random() for i in range(100000)]
x_train = numpy.array(x, dtype=numpy.float32)
x_train = x_train.reshape(-1, 1)
y = [2 * i + 1 for i in x]
y_train = numpy.array(y, dtype=numpy.float32)
y_train = y_train.reshape(-1, 1)

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

epochs = 3000
learning_rate = 0.01
optimzer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

if __name__ == "__main__":
    for epoch in range(epochs):
        epoch += 1

        inputs = torch.from_numpy(x_train)
        labels = torch.from_numpy(y_train)

        optimzer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimzer.step()
        if epoch % 50 == 0:
            print('epoch {},loss {}'.format(epoch, loss.item()))

    torch.save(model, "m.pt")
