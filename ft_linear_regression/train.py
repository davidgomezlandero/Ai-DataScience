import numpy as np
import warnings
import matplotlib.pyplot as plot


try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = np.loadtxt('data.csv', delimiter=',', skiprows= 1)
except (OSError, IOError):
    print("Error: data.csv is empty of has invalid format.")
    exit(1)
except (StopIteration, ValueError, UserWarning):
	print("Error: data.csv is empty or has invalid format.")
	exit(1)

if data.size == 0:
    print("Error: data.csv contains no data")
    
if np.any(data < 0) or np.any(data > 1000000):
    print("Error: All values must be between 0 and 1,000,000.")
    exit(1)
    
tmp_theta0 = 0
tmp_theta1 = 0
mileage = data[:,0]
price = data[:,1]
mileage_mean = np.mean(mileage)
mileage_std = np.std(mileage)
mileage_norm = (mileage - mileage_mean) / mileage_std

epochs = 200
learning_rate = 0.02
loss_list = []

def predict(mileage):
    return tmp_theta0 + tmp_theta1 * mileage

def compute_loss():
    total = 0.0
    for i in range(len(mileage_norm)):
        error = predict(mileage_norm[i]) - price[i]
        total += error * error
    return total / (2 * len(mileage_norm))


for _ in range(epochs):
    sum0 = 0
    sum1 = 0
    for i in range(len(mileage_norm)):
        sum0 += predict(mileage_norm[i]) - price[i]
        sum1 += (predict(mileage_norm[i]) - price[i])* mileage_norm[i]
    tmp_theta0 -= learning_rate * (1 / len(mileage_norm)) * sum0
    tmp_theta1 -= learning_rate * (1 / len(mileage_norm)) * sum1
    loss = compute_loss()
    loss_list.append(loss)


theta1 = tmp_theta1 / mileage_std
theta0 = tmp_theta0 - theta1 * mileage_mean

x1 = mileage
y1 = [theta0 + theta1 *x for x in x1]



plot.scatter(mileage, price, color = "blue", label = "Points")
plot.title("Points")
plot.xlabel("Mileage")
plot.ylabel("Price")
plot.legend()
plot.grid(True)

plot.savefig("points_data.png")
plot.close()

plot.plot(x1, y1, label=f'y = {round(theta0,4)} + {round(theta1,4)} * x', color="blue")
plot.scatter(mileage, price, color="red", label="Puntos")
plot.title("Recta y puntos")
plot.xlabel("x")
plot.ylabel("y")
plot.legend()
plot.grid(True)

plot.savefig("grafica1.png")
plot.close()

plot.plot(range(len(loss_list)), loss_list)
plot.xlabel("Epoch")
plot.ylabel("Loss (MSE)")
plot.title("Loss vs Epochs")
plot.grid(True)
plot.savefig("loss_curve.png")
plot.close()

with open("theta.csv", "w") as f:
    f.write("theta0,theta1\n")
    f.write(f"{theta0},{theta1}\n")