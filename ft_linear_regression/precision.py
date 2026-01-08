import math
import numpy as np
import warnings


""" Parsing the input data """

try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = np.loadtxt('data.csv', delimiter=',', skiprows= 1)
except (OSError, IOError):
    print("Error: data.csv is empty or has invalid format.")
    exit(1)
except (StopIteration, ValueError, UserWarning):
	print("Error: data.csv is empty or has invalid format.")
	exit(1)

if data.size == 0:
    print("Error: data.csv contains no data")
    
if np.any(data < 0) or np.any(data > 1000000):
    print("Error: All values must be between 0 and 1,000,000.")
    exit(1)

""" Loading theta0 and theta1 """
 
def load_theta(filename = "theta.csv"):
    try:
        with open(filename, "r") as f:
            next(f)
            line = f.readline()
            t0,t1 = line.strip().split(",")
            return float(t0), float(t1)
    except (FileNotFoundError, ValueError):
        return 0.0, 0.0

theta0,theta1 = load_theta()

if theta0 == 0.0 and theta1 == 0.0:
    print("Warning: Model not trained yet. Using default values (0, 0)")

mileage = data[:,0]
price = data[:,1]

""" Function of prediction """

def estimatePrice(mileage) : return theta0 + (theta1 * mileage)

""" R²(Coefficient of determination) """

price_mean = sum(price) / len(price)
tss = sum((y - price_mean) ** 2 for y in price)
rss = sum((price[i] - estimatePrice(mileage[i])) ** 2 for i in range(len(price)))


r2 = 1 - (rss/tss)
print(f'The coefficient of determination for this linear regression is: {round(r2,4)}')
print(f'This means the model explains {round(r2*100, 2)}% of the variance in car prices based on mileage.')
