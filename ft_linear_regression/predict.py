
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

if theta0 == 0.0 and theta1 == 0.0:
    print("Warning: Model not trained yet. Using default values (0, 0)")

def estimatePrice(mileage) : return theta0 + (theta1 * mileage)

try:
    mileage = float(input("Mileage to predict the price:\n"))
    if mileage < 0 or mileage > 400000:
        print("Error: Mileage cannot be negative and must be lower than 400.000km to avoid negative prices")
        exit(1)
except ValueError:
    print("Error: Invalid input. Please enter a number.")
    exit(1)

print(f'The predicted price for {mileage}km is {round(estimatePrice(mileage), 2)}€')