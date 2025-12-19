theta0 = 0
theta1 = 0

def estimatePrice(mileage) : theta0 + (theta1 * mileage)

mileage = input("Mileage to predict the price")

print(f'The predicted price for {mileage}km is {estimatePrice(mileage)}€')