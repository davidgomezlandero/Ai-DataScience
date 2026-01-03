
import math

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

def estimatePrice(mileage) : return theta0 + (theta1 * mileage)

mileage = float(input("Mileage to predict the price:\n"))



print(f'The predicted price for {mileage}km is {math.ceil(estimatePrice(mileage) * 100) / 100}€')