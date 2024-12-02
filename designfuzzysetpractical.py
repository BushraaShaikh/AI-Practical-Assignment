"""#Here's a simple example of designing a fuzzy set in Python using the scikit-fuzzy library.
#We'll design a fuzzy set to evaluate the quality of a service at a restaurant, where the service quality is described as "poor," "average," or "good."
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define the universe of discourse for service quality
x_service = np.arange(0, 11, 1)  # Service quality ranges from 0 to 10

# Define fuzzy sets for service quality
service_poor = fuzz.trapmf(x_service, [0, 0, 2, 4])  # Trapezoidal
service_average = fuzz.trimf(x_service, [2, 5, 8])  # Triangular
service_good = fuzz.trapmf(x_service, [6, 8, 10, 10])  # Trapezoidal

# Plot the fuzzy sets
plt.figure(figsize=(8, 5))
plt.plot(x_service, service_poor, label="Poor", linewidth=2)
plt.plot(x_service, service_average, label="Average", linewidth=2)
plt.plot(x_service, service_good, label="Good", linewidth=2)

# Add labels, legend, and title
plt.title("Fuzzy Sets for Service Quality")
plt.xlabel("Service Quality")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid(True)
plt.show()"""
"""1.	Explanation:
1.	Universe of Discourse:
•	The variable x_service represents service quality, which ranges from 0 (worst) to 10 (best).
2.	Fuzzy Sets:
•	Poor: A trapezoidal membership function with full membership between 0 and 2, gradually decreasing from 2 to 4.
•	Average: A triangular membership function peaking at 5 and covering the range from 2 to 8.
•	Good: A trapezoidal membership function with full membership between 8 and 10, gradually increasing from 6 to 8.
3.	Visualization:
•	The plot shows how the membership degree varies with the service quality score.
________________________________________
Output:
Running the code will generate a plot showing the fuzzy sets for "poor," "average," and "good" service quality.
This example demonstrates how to design and visualize fuzzy sets, which can be used in fuzzy logic systems for decision-making.
 
"""





"""Designing a fuzzy set for shape matching of handwritten characters involves defining membership functions that measure the similarity between a given shape and the ideal features of a handwritten character. Here's a Python implementation that uses the fuzzy logic library scikit-fuzzy for this purpose:
Steps:
1.	Define input parameters: Determine the key features of a handwritten character (e.g., aspect ratio, curvature, line intersections).
2.	Create membership functions: Use fuzzy sets to represent the degree of membership for each feature.
3.	Aggregate results: Combine the fuzzy scores for all features to assess the overall match."""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
aspect_ratio = ctrl.Antecedent(np.arange(0.5, 2.5, 0.1), 'aspect_ratio')  # Example range
curvature = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'curvature')          # Example range
line_intersections = ctrl.Antecedent(np.arange(0, 5, 1), 'line_intersections')  # Example range
shape_match = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'shape_match')      # Degree of match

# Membership functions for aspect_ratio
aspect_ratio['small'] = fuzz.trapmf(aspect_ratio.universe, [0.5, 0.5, 0.9, 1.1])
aspect_ratio['medium'] = fuzz.trimf(aspect_ratio.universe, [0.9, 1.2, 1.5])
aspect_ratio['large'] = fuzz.trapmf(aspect_ratio.universe, [1.3, 1.5, 2.5, 2.5])

# Membership functions for curvature
curvature['low'] = fuzz.trapmf(curvature.universe, [0, 0, 0.3, 0.5])
curvature['medium'] = fuzz.trimf(curvature.universe, [0.3, 0.5, 0.7])
curvature['high'] = fuzz.trapmf(curvature.universe, [0.5, 0.7, 1.0, 1.0])

# Membership functions for line_intersections
line_intersections['few'] = fuzz.trapmf(line_intersections.universe, [0, 0, 1, 2])
line_intersections['moderate'] = fuzz.trimf(line_intersections.universe, [1, 2, 3])
line_intersections['many'] = fuzz.trapmf(line_intersections.universe, [2, 3, 5, 5])

# Membership functions for shape_match
shape_match['poor'] = fuzz.trapmf(shape_match.universe, [0, 0, 0.3, 0.5])
shape_match['average'] = fuzz.trimf(shape_match.universe, [0.3, 0.5, 0.7])
shape_match['good'] = fuzz.trapmf(shape_match.universe, [0.5, 0.7, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(aspect_ratio['small'] & curvature['low'], shape_match['poor'])
rule2 = ctrl.Rule(aspect_ratio['medium'] & curvature['medium'], shape_match['average'])
rule3 = ctrl.Rule(aspect_ratio['large'] & curvature['high'] & line_intersections['many'], shape_match['good'])

# Combine rules into a control system
shape_matching_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
shape_matching_sim = ctrl.ControlSystemSimulation(shape_matching_ctrl)

# Input values for a specific character
shape_matching_sim.input['aspect_ratio'] = 1.2
shape_matching_sim.input['curvature'] = 0.6
shape_matching_sim.input['line_intersections'] = 2

# Compute the result
shape_matching_sim.compute()

# Output the degree of shape match
print(f"Shape Match Score: {shape_matching_sim.output['shape_match']:.2f}")

"""
Explanation:
1.	Input Features:
•	aspect_ratio: Ratio of height to width of the character.
•	curvature: Degree of curve in the strokes.
•	line_intersections: Number of intersections among strokes.
2.	Fuzzy Membership Functions:
•	Define the linguistic categories (e.g., small, medium, large) for each feature.
3.	Fuzzy Rules:
•	Logical conditions to determine the output (shape matching score) based on input features.
4.	Result:
•	The final output is a fuzzy score that indicates how well the input character matches the ideal shape.
This approach can be extended by refining the membership functions, adding more features, and adjusting rules to better suit the dataset of handwritten characters.





"""


