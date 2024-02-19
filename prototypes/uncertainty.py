import numpy as np
import matplotlib.pyplot as plt

# Logistic function for probability
def logistic_function(x, L, k, x_0):
    return L / (1 + np.exp(-k * (x - x_0)))

def inverted_logistic_function(x, L, k, x_0):
    return L - L / (1 + np.exp(-k * (x - x_0)))

def update_k_based_on_turn_rate(turn_rate, k_max, k_min, max_turn_rate, confidence=0.95):
    """
    Update the steepness parameter k based on the turn rate.

    :param turn_rate: Current turn rate
    :param k_max: Maximum value of k (for the highest turn rate)
    :param k_min: Minimum value of k (for no turn or very slow turn rate)
    :param max_turn_rate: Maximum possible turn rate
    :param confidence: Confidence level for accuracy of the turn rate
    :return: Updated value of k
    """
    # Ensure turn_rate does not exceed max_turn_rate
    turn_rate = min(turn_rate, max_turn_rate)
    
    # Calculate new k value based on turn rate
    k = k_min + (1 - (turn_rate / max_turn_rate)) * (k_max - k_min) * confidence

    return k

# Parameters for the probability model
L = 1      # Maximum probability
k = 0.3    # Steepness of the curve, adjust as needed
x_0 = 23   # Midpoint of the curve, average velocity

maximum_turn_rate = np.deg2rad(30)  # 30 degrees per second
k_max = 0.9
k_min = 0.1

# Velocity range for plotting
velocity_range = np.linspace(0, 50, 400)  # Covering a broader range around the specified velocity

#turn_rates = np.linspace(0, maximum_turn_rate, np.deg2rad(1))  # 1 degree per second
turn_rates = np.linspace(0, maximum_turn_rate, 10)  # 10 degree per second

#formulate k based on its turning fast or slow
# if straight line k becomes 

# Calculating the probability
probabilities = []
for turn in turn_rates:
    k = update_k_based_on_turn_rate(turn, k_max, k_min, maximum_turn_rate)
    probability = logistic_function(velocity_range, L, k, x_0)
    # probability = inverted_logistic_function(velocity_range, L, k, x_0)
    probabilities.append(probability)

# Plotting the probability model
plt.figure(figsize=(10, 6))

for i, probability in enumerate(probabilities):
    plt.plot(velocity_range, probability, 
             label='turn rate = {}'.format(np.rad2deg(turn_rates[i])))
#plt.plot(velocity_range, probability, label='Probability of Velocity (25 ± 10)', color='green')
plt.title('Probability Distribution of Velocity as an S-Curve')
plt.xlabel('Velocity')
plt.ylabel('Probability')
plt.ylim(0, 1)  # Setting y-axis limits from 0 to 1
plt.legend()
plt.grid(True)
plt.show()
