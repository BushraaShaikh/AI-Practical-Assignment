#A.I.
#1 Direct Heuristic Search Techniques
import heapq

class HeuristicSearch:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, current):
        """Manhattan distance heuristic."""
        return abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1])

    def is_valid(self, x, y):
        """Check if a cell is valid and within grid bounds."""
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0

    def best_first_search(self):
        """Implements Best-First Search."""
        open_set = [(self.heuristic(self.start), self.start)]
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == self.goal:
                return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and self.is_valid(*neighbor):
                    heapq.heappush(open_set, (self.heuristic(neighbor), neighbor))

        return False

    def greedy_search(self):
        """Implements Greedy Search."""
        open_set = [(self.heuristic(self.start), self.start)]
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == self.goal:
                return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and self.is_valid(*neighbor):
                    heapq.heappush(open_set, (self.heuristic(neighbor), neighbor))

        return False

    def hill_climbing(self):
        """Implements Hill-Climbing."""
        current = self.start

        while current != self.goal:
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.is_valid(*neighbor):
                    neighbors.append((self.heuristic(neighbor), neighbor))

            if not neighbors:
                return False  # No path found

            # Choose the neighbor with the smallest heuristic
            next_move = min(neighbors, key=lambda x: x[0])[1]

            # If the heuristic doesn't improve, stop
            if self.heuristic(next_move) >= self.heuristic(current):
                return False

            current = next_move

        return True

# Example Usage
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)

    search = HeuristicSearch(grid, start, goal)
    print("Best-First Search:", search.best_first_search())
    print("Greedy Search:", search.greedy_search())
    print("Hill-Climbing:", search.hill_climbing())

"""OUTPUT:
Best-First Search: True
Greedy Search: True
Hill-Climbing: True"""


#2 AOSTAR ALGORITHM
class AOStar:
    def __init__(self, graph, heuristic):
        """
        Initialize the AO* algorithm.
        :param graph: A dictionary where each node maps to its children (AND/OR arcs).
        :param heuristic: A dictionary of heuristic values for each node.
        """
        self.graph = graph
        self.heuristic = heuristic
        self.solution_graph = {}

    def explore_and_backtrack(self, node):
        """
        Explore the graph recursively and perform backtracking to update costs.
        :param node: The current node being processed.
        :return: The updated cost for the current node.
        """
        # If the node is a terminal node, return its heuristic cost
        if node not in self.graph or not self.graph[node]:
            self.solution_graph[node] = None
            return self.heuristic[node]

        min_cost = float('inf')
        best_option = None

        # Explore all arcs (AND/OR) for the current node
        for children in self.graph[node]:
            cost = 0
            for child in children:
                cost += self.explore_and_backtrack(child)  # Recursive call

            # Choose the arc with the minimum cost
            if cost < min_cost:
                min_cost = cost
                best_option = children

        # Update the solution graph and heuristic cost
        self.solution_graph[node] = best_option
        self.heuristic[node] = min_cost

        return min_cost

    def find_solution(self, start_node):
        """
        Find the solution starting from the root node.
        :param start_node: The root node of the graph.
        :return: The final solution graph.
        """
        self.explore_and_backtrack(start_node)
        return self.solution_graph

# Example Usage
if __name__ == "__main__":
    # Define an AND-OR graph as a dictionary
    # Each node maps to a list of arcs, where each arc is a list of child nodes
    graph = {
        'A': [['B', 'C'], ['D']],
        'B': [['E'], ['F']],
        'C': [['G']],
        'D': [['H']],
        'E': [],
        'F': [],
        'G': [],
        'H': []
    }

    # Define heuristic values for each node
    heuristic = {
        'A': 1, 'B': 1, 'C': 2, 'D': 2,
        'E': 3, 'F': 4, 'G': 5, 'H': 6
    }

    # Run AO* algorithm
    aostar = AOStar(graph, heuristic)
    solution = aostar.find_solution('A')

    print("Solution Graph:", solution)

"""OUTPUT:
Solution Graph: {'E': None, 'F': None, 'B': ['E'], 'G': None, 'C': ['G'], 'H': None, 'D': ['H'], 'A': ['D']}
"""

#2 implementation of A* Algorithm

def A_star(start_node,stop_node):
    open_set=set(start_node)
    close_set=set()
    g={}  #store distance from starting node
    parents={}  #contains an adjency map of all nodes
    g[start_node]=0
    parents[start_node]=start_node  #if start node is root node i.e. it has no parent node so start node is set to its own parent node
    while len(open_set)>0:
        n=None
        #node with lowest F(n) is found
        for v in open_set:
            if n==None or g[v]+heuristic(v)<g[n]+heuristic(n):
                n=v
        if n==stop_node or graph_nodes[n]==None:
            pass
        else:
            for (m,weight) in get_neighboures(n):
            #nodes m not in first and last set are added and n is set as its parent
                if m not in open_set and m not in close_set:
                    open_set.add(m)
                    parents[m]=n
                    g[m]=g[n]+weight
                else:  
                #for each node m compare its distance from start to n node
                    if g[m]>g[n]+weight:
                        g[m]=g[n]+weight  #update value of g[m]
                        parents[m]=n  #change parent of m to n
                        if m in close_set:  #if m is present in close set remove and add to open set
                            close_set.remove(m)
                            open_set.add(m)
        if n==None:
            print("Path does not exist..!")
            return None
        if n==stop_node:
        #if current node is the stop node then we begin reconstructing the path from it to the start node
            path=[]
            while parents[n]!=n:
                path.append(n)
                n=parents[n]
            path.append(start_node)
            path.reverse()
            print("Path found:{}".format(path))
            return path
    #remove n from the open list and add into the close list because all its neighbours are inspected
        open_set.remove(n)
        close_set.add(n)
    print("Path does not exist")
    return None
    
#define function to return neighbours and its distance from the pass nodes
def get_neighboures(v):
    if v in graph_nodes:
        return graph_nodes[v]
    else:
        return None
        
def heuristic(n):
#consider heuristic distances given and its functions for all nodes
    h_dist={'A':11, 'B':6, 'C':19, 'D':1, 'E':7, 'G':0}
    return h_dist[n]
    
graph_nodes={
    'A':[('B',2),('E',5)], 
    'B':[('C',1),('G',9)], 
    'C':'None', 
    'D':[('G',1)], 
    'E':[('D',6)]
    }

A_star('A','G')

"""OUTPUT:
Path found:['A', 'B', 'G']
"""

#2 BEST FIRST SEARCH ALGORITHM
#This function will visit all nodes of graph using bfs traversal
def bfs_connected_component(graph,start):
    explored=[]  #keep track of all visited nodes
    queue=[start]  #keep track of nodes to be check
    
    while queue:  #keep looping until their are nodes still to be check
        node=queue.pop(0)  #pop shallowest node/first node from queue
        
        if node not in explored:
            explored.append(node)  #add node to list of checked nodes
            neighbours=graph[node]  
            
            for neighbour in neighbours:
                queue.append(neighbour)
    return explored

#this function will find the shortest path between two nodes of a graph using bfs
def bfs_shortest_path(graph,start,goal):
    explored=[]
    queue=[[start]]
    
    if start==goal:
        return "That was easy because the start node is goal node"
    
    while queue:
        path=queue.pop(0)  #pop the first path from the queue
        node=path[-1]  #it will return last node from the path

        if node not in explored:
            neighbours=graph[node]
            #go through all neighbour nodes, construct a new path and push it into the queue

            for neighbour in neighbours:
                new_path=list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour==goal:
                    return new_path  #return path if neighbour is the goal node
            
            explored.append(node)    #mark node as explored
            
    return "Path does not exist"    #in case if there is no path between the two nodes
    
if __name__=='__main__':
    graph={'A':['B','C'], 'B':['A','D'], 'C':['A','E','F'], 'D':['B'], 'E':['C'], 'F':['C']}
    print("Here are the nodes of the graph visited by BFS starting from node A: ",bfs_connected_component(graph,'A'))
    print("Here is the shortest path between nodes D and E: ",bfs_shortest_path(graph,'D','E'))
    
"""OUTPUT:
Here are the nodes of the graph visited by BFS starting from node A:  ['A', 'B', 'C', 'D', 'E', 'F']
Here is the shortest path between nodes D and E:  ['D', 'B', 'A', 'C', 'E']
"""
#2 HILL CLIMBING ALGORITHM
import random

class HillClimbing:
    def __init__(self, objective_function, neighbors_function):
        """
        Initialize the Hill Climbing algorithm.
        :param objective_function: Function to evaluate the fitness of a solution.
        :param neighbors_function: Function to generate neighbors for a given solution.
        """
        self.objective_function = objective_function
        self.neighbors_function = neighbors_function

    def climb(self, start, max_iterations=1000):
        """
        Perform the Hill Climbing algorithm.
        :param start: The starting solution.
        :param max_iterations: Maximum number of iterations.
        :return: The best solution found and its objective value.
        """
        current_solution = start
        current_value = self.objective_function(current_solution)
        
        for _ in range(max_iterations):
            neighbors = self.neighbors_function(current_solution)
            if not neighbors:
                break  # No neighbors to explore
            
            next_solution = max(neighbors, key=self.objective_function)
            next_value = self.objective_function(next_solution)
            
            # Stop if no improvement
            if next_value <= current_value:
                break
            
            current_solution, current_value = next_solution, next_value

        return current_solution, current_value

# Example Usage
if __name__ == "__main__":
    # Example: Maximize the objective function f(x) = -x^2 + 5x + 10

    def objective_function(x):
        # The objective function to maximize
        return -x**2 + 5 * x + 10

    def neighbors_function(x):
        # Generate neighbors by moving +/- 1 step
        step = 1
        return [x - step, x + step]

    # Starting point
    start = random.randint(-10, 10)

    # Run the algorithm
    hill_climbing = HillClimbing(objective_function, neighbors_function)
    best_solution, best_value = hill_climbing.climb(start)

    print("Best Solution:", best_solution)
    print("Objective Value at Best Solution:", best_value)

"""OUTPUT:
Best Solution: 2
Objective Value at Best Solution: 16
"""

#3 PERCEPTRON LEARNING ALGORITHM
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Activation function: Step function"""
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        """
        Train the Perceptron algorithm.
        Args:
        - X: Input features (numpy array of shape [n_samples, n_features]).
        - y: Labels (numpy array of shape [n_samples]).
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Linear combination of weights and input features
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)

                # Update weights and bias if prediction is incorrect
                error = y[idx] - y_pred
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

    def predict(self, X):
        """
        Make predictions using the trained model.
        Args:
        - X: Input features (numpy array of shape [n_samples, n_features]).
        Returns:
        - Predictions (numpy array of shape [n_samples]).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

# Example Usage
if __name__ == "__main__":
    # AND gate data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])  # AND gate output

    # Initialize and train the perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)

    # Test the perceptron
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
    print("Weights:", perceptron.weights)
    print("Bias:", perceptron.bias)

"""OUTPUT:
Predictions: [0 0 0 1]
Weights: [0.2 0.1]
Bias: -0.20000000000000004
"""

#4 REAL LIFE APPLICATION IN AI LIBRARIES PYTHON
#Application: Sentiment Analysis of Customer Reviews
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalysis:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.vectorizer = None

    def preprocess(self, text):
        """Clean and preprocess text data."""
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        words = word_tokenize(text)  # Tokenize
        words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
        return ' '.join(words)

    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        # Load dataset
        data = pd.read_csv(self.dataset_path)

        # Preprocess text
        data['review'] = data['review'].apply(self.preprocess)

        # Convert labels to binary (positive = 1, negative = 0)
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

        # Split data into features and labels
        X = data['review']
        y = data['sentiment']

        return X, y

    def train_model(self, X, y):
        """Train the sentiment analysis model."""
        # Convert text data to feature vectors
        self.vectorizer = CountVectorizer(max_features=1000)
        X_vectorized = self.vectorizer.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42
        )

        # Train a Naive Bayes classifier
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, review):
        """Predict sentiment for a given review."""
        if not self.model or not self.vectorizer:
            raise Exception("Model not trained. Call train_model() first.")

        # Preprocess and vectorize the input review
        review = self.preprocess(review)
        review_vectorized = self.vectorizer.transform([review])

        # Predict sentiment
        prediction = self.model.predict(review_vectorized)
        return "Positive" if prediction[0] == 1 else "Negative"

# Example Usage
if __name__ == "__main__":
    # Create a dataset file with sample reviews for testing
    sample_data = {
        'review': [
            "The product is great and works perfectly!",
            "Terrible experience, I want a refund.",
            "Customer service was helpful and quick.",
            "The item broke after two uses. Awful!",
            "Absolutely love this! Highly recommend.",
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    dataset_path = 'customer_reviews.csv'
    pd.DataFrame(sample_data).to_csv(dataset_path, index=False)

    # Initialize and train sentiment analysis model
    sentiment_analyzer = SentimentAnalysis(dataset_path)
    X, y = sentiment_analyzer.load_and_prepare_data()
    sentiment_analyzer.train_model(X, y)

    # Test the model with a custom review
    test_review = "The delivery was quick but the product quality is poor."
    print(f"Review: {test_review}")
    print("Predicted Sentiment:", sentiment_analyzer.predict(test_review))

"""OUTPUT:
Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00       1.0
           1       0.00      0.00      0.00       0.0

    accuracy                           0.00       1.0
   macro avg       0.00      0.00      0.00       1.0
weighted avg       0.00      0.00      0.00       1.0

Review: The delivery was quick but the product quality is poor.
Predicted Sentiment: Positive
"""

#5 EXPERT SYSTEM IN PYTHON
#Code: Expert System for Diagnosing Illnesses
class ExpertSystem:
    def __init__(self):
        self.knowledge_base = {
            "common_cold": {
                "symptoms": ["cough", "sneezing", "sore throat", "runny nose", "congestion"],
                "description": "You may have a common cold. It is usually caused by a viral infection.",
                "advice": "Rest, drink fluids, and consider over-the-counter cold medications."
            },
            "flu": {
                "symptoms": ["fever", "chills", "body aches", "fatigue", "headache"],
                "description": "You may have the flu. It is caused by the influenza virus.",
                "advice": "Rest, stay hydrated, and consider consulting a doctor if symptoms persist."
            },
            "allergy": {
                "symptoms": ["sneezing", "runny nose", "itchy eyes", "congestion"],
                "description": "You may have allergies. Allergies are caused by a reaction to allergens like pollen or dust.",
                "advice": "Avoid allergens, consider antihistamines, and consult an allergist if necessary."
            },
            "migraine": {
                "symptoms": ["headache", "nausea", "sensitivity to light", "sensitivity to sound"],
                "description": "You may have a migraine. It is a neurological condition that can cause severe headaches.",
                "advice": "Rest in a dark, quiet room, and consider pain relievers or migraine-specific medication."
            }
        }

    def diagnose(self, symptoms):
        """Diagnoses the condition based on the symptoms provided."""
        for condition, details in self.knowledge_base.items():
            if all(symptom in symptoms for symptom in details["symptoms"]):
                return {
                    "condition": condition,
                    "description": details["description"],
                    "advice": details["advice"]
                }
        return {
            "condition": "unknown",
            "description": "Your symptoms do not match any condition in our database.",
            "advice": "Consult a healthcare professional for further assistance."
        }

# Example Usage
if __name__ == "__main__":
    print("Welcome to the Health Expert System!")
    print("Please enter your symptoms separated by commas.")
    user_input = input("Symptoms: ").strip().lower()
    user_symptoms = [symptom.strip() for symptom in user_input.split(",")]

    expert_system = ExpertSystem()
    diagnosis = expert_system.diagnose(user_symptoms)

    print("\nDiagnosis:")
    print(f"Condition: {diagnosis['condition']}")
    print(f"Description: {diagnosis['description']}")
    print(f"Advice: {diagnosis['advice']}")

"""OUTPUT:
Welcome to the Health Expert System!
Please enter your symptoms separated by commas.
Symptoms: common_cold

Diagnosis:
Condition: unknown
Description: Your symptoms do not match any condition in our database.
Advice: Consult a healthcare professional for further assistance.
"""
#5 expert system in python
class ExpertSystem:
    def __init__(self):
        self.knowledge_base = {
            "common_cold": {
                "symptoms": ["cough", "sneezing", "sore throat", "runny nose", "congestion"],
                "description": "You may have a common cold. It is usually caused by a viral infection.",
                "advice": "Rest, drink fluids, and consider over-the-counter cold medications."
            },
            "flu": {
                "symptoms": ["fever", "chills", "body aches", "fatigue", "headache"],
                "description": "You may have the flu. It is caused by the influenza virus.",
                "advice": "Rest, stay hydrated, and consider consulting a doctor if symptoms persist."
            },
            "allergy": {
                "symptoms": ["sneezing", "runny nose", "itchy eyes", "congestion"],
                "description": "You may have allergies. Allergies are caused by a reaction to allergens like pollen or dust.",
                "advice": "Avoid allergens, consider antihistamines, and consult an allergist if necessary."
            },
            "migraine": {
                "symptoms": ["headache", "nausea", "sensitivity to light", "sensitivity to sound"],
                "description": "You may have a migraine. It is a neurological condition that can cause severe headaches.",
                "advice": "Rest in a dark, quiet room, and consider pain relievers or migraine-specific medication."
            }
        }

    def diagnose(self, symptoms):
        """Diagnoses the condition based on the symptoms provided."""
        for condition, details in self.knowledge_base.items():
            if any(symptom in symptoms for symptom in details["symptoms"]):  # Check if any symptom matches
                return {
                    "condition": condition,
                    "description": details["description"],
                    "advice": details["advice"]
                }
        return {
            "condition": "unknown",
            "description": "Your symptoms do not match any condition in our database.",
            "advice": "Consult a healthcare professional for further assistance."
        }

# Example Usage
if __name__ == "__main__":
    print("Welcome to the Health Expert System!")
    print("Please enter your symptoms separated by commas.")
    user_input = input("Symptoms: ").strip().lower()
    user_symptoms = [symptom.strip() for symptom in user_input.split(",")]

    expert_system = ExpertSystem()
    diagnosis = expert_system.diagnose(user_symptoms)

    print("\nDiagnosis:")
    print(f"Condition: {diagnosis['condition']}")
    print(f"Description: {diagnosis['description']}")
    print(f"Advice: {diagnosis['advice']}")


"""output:
Welcome to the Health Expert System!
Please enter your symptoms separated by commas.
Symptoms: sneezing, runny nose, itchy eyes, congestion

Diagnosis:
Condition: common_cold
Description: You may have a common cold. It is usually caused by a viral infection.
Advice: Rest, drink fluids, and consider over-the-counter cold medications.
"""

#6 two player game using min max search algorithm
import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # 3x3 board
        self.current_winner = None  # Track the winner

    def print_board(self):
        """Prints the current board state."""
        for row in [self.board[i * 3:(i + 1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        """Returns a list of available positions."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        """Checks if there are empty squares."""
        return ' ' in self.board

    def num_empty_squares(self):
        """Returns the number of empty squares."""
        return self.board.count(' ')

    def make_move(self, square, letter):
        """
        Makes a move on the board.
        Args:
        - square: Position to place the move.
        - letter: 'X' or 'O'.
        """
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        """
        Checks if there is a winner.
        Args:
        - square: Last move position.
        - letter: 'X' or 'O'.
        """
        # Check row
        row_ind = square // 3
        row = self.board[row_ind * 3:(row_ind + 1) * 3]
        if all([spot == letter for spot in row]):
            return True

        # Check column
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True

        # Check diagonals
        if square % 2 == 0:  # Only possible for even-indexed squares
            diagonal1 = [self.board[i] for i in [0, 4, 8]]  # Top-left to bottom-right
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]  # Top-right to bottom-left
            if all([spot == letter for spot in diagonal2]):
                return True

        return False

def minimax(state, player, alpha, beta, depth=0):
    """
    Minimax algorithm with alpha-beta pruning.
    Args:
    - state: Current board state.
    - player: 'X' (maximizer) or 'O' (minimizer).
    - alpha: Best value for maximizer.
    - beta: Best value for minimizer.
    - depth: Current depth of recursion.
    """
    max_player = 'X'  # Human
    other_player = 'O' if player == 'X' else 'X'

    # Check for a terminal state
    if state.current_winner == other_player:
        return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (state.num_empty_squares() + 1)}

    elif not state.empty_squares():  # No empty squares
        return {'position': None, 'score': 0}

    if player == max_player:  # Maximizing player
        best = {'position': None, 'score': -math.inf}
    else:  # Minimizing player
        best = {'position': None, 'score': math.inf}

    for possible_move in state.available_moves():
        # Make the move
        state.make_move(possible_move, player)
        sim_score = minimax(state, other_player, alpha, beta, depth + 1)  # Simulate the game

        # Undo the move
        state.board[possible_move] = ' '
        state.current_winner = None
        sim_score['position'] = possible_move

        # Update the best move
        if player == max_player:
            if sim_score['score'] > best['score']:
                best = sim_score
            alpha = max(alpha, best['score'])
        else:
            if sim_score['score'] < best['score']:
                best = sim_score
            beta = min(beta, best['score'])

        if beta <= alpha:
            break

    return best

def play():
    game = TicTacToe()
    print("Welcome to Tic Tac Toe!")
    game.print_board()

    while game.empty_squares():
        # Human move
        if game.num_empty_squares() % 2 == 0:
            while True:  # Loop until a valid move is made
                try:
                    square = int(input("Enter your move (0-8): "))
                    if square < 0 or square > 8:
                        print("Please enter a valid number between 0 and 8.")
                        continue
                    if game.board[square] != ' ':
                        print("This position is already occupied. Try again.")
                        continue
                    if game.make_move(square, 'X'):
                        break
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 8.")
        else:  # AI move
            print("AI is thinking...")
            move = minimax(game, 'O', -math.inf, math.inf)
            game.make_move(move['position'], 'O')

        game.print_board()

        if game.current_winner:
            if game.current_winner == 'X':
                print("You win!")
            else:
                print("AI wins!")
            return

    print("It's a tie!")

if __name__ == "__main__":
    play()

"""output:
Welcome to Tic Tac Toe!
|   |   |   |
|   |   |   |
|   |   |   |
AI is thinking...
| O |   |   |
|   |   |   |
|   |   |   |
Enter your move (0-8): 0
This position is already occupied. Try again.
Enter your move (0-8): 4
| O |   |   |
|   | X |   |
|   |   |   |
AI is thinking...
| O | O |   |
|   | X |   |
|   |   |   |
Enter your move (0-8): 2
| O | O | X |
|   | X |   |
|   |   |   |
AI is thinking...
| O | O | X |
|   | X |   |
| O |   |   |
Enter your move (0-8): 3
| O | O | X |
| X | X |   |
| O |   |   |
AI is thinking...
| O | O | X |
| X | X | O |
| O |   |   |
Enter your move (0-8): 7
| O | O | X |
| X | X | O |
| O | X |   |
AI is thinking...
| O | O | X |
| X | X | O |
| O | X | O |
It's a tie!
"""

#7 fuzzy set for shape matching of hand written character
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class HandwrittenCharacterMatcher:
    def __init__(self):
        self.attributes = {
            "curvature": np.arange(0, 101, 1),  # Curvature values (0 to 100)
            "angle": np.arange(0, 181, 1),  # Angle values (0° to 180°)
            "stroke_length": np.arange(0, 51, 1),  # Stroke length (0 to 50)
        }

        # Define fuzzy membership functions for each attribute
        self.curvature_membership = {
            "low": fuzz.trimf(self.attributes["curvature"], [0, 0, 50]),
            "medium": fuzz.trimf(self.attributes["curvature"], [25, 50, 75]),
            "high": fuzz.trimf(self.attributes["curvature"], [50, 100, 100]),
        }

        self.angle_membership = {
            "small": fuzz.trimf(self.attributes["angle"], [0, 0, 90]),
            "medium": fuzz.trimf(self.attributes["angle"], [45, 90, 135]),
            "large": fuzz.trimf(self.attributes["angle"], [90, 180, 180]),
        }

        self.stroke_length_membership = {
            "short": fuzz.trimf(self.attributes["stroke_length"], [0, 0, 25]),
            "medium": fuzz.trimf(self.attributes["stroke_length"], [10, 25, 40]),
            "long": fuzz.trimf(self.attributes["stroke_length"], [25, 50, 50]),
        }

    def visualize_memberships(self):
        """Visualizes the fuzzy membership functions."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Curvature
        axes[0].plot(
            self.attributes["curvature"],
            self.curvature_membership["low"],
            label="Low",
            color="blue"
        )
        axes[0].plot(
            self.attributes["curvature"],
            self.curvature_membership["medium"],
            label="Medium",
            color="green"
        )
        axes[0].plot(
            self.attributes["curvature"],
            self.curvature_membership["high"],
            label="High",
            color="red"
        )
        axes[0].set_title("Curvature")
        axes[0].legend()

        # Angle
        axes[1].plot(
            self.attributes["angle"],
            self.angle_membership["small"],
            label="Small",
            color="blue"
        )
        axes[1].plot(
            self.attributes["angle"],
            self.angle_membership["medium"],
            label="Medium",
            color="green"
        )
        axes[1].plot(
            self.attributes["angle"],
            self.angle_membership["large"],
            label="Large",
            color="red"
        )
        axes[1].set_title("Angle")
        axes[1].legend()

        # Stroke Length
        axes[2].plot(
            self.attributes["stroke_length"],
            self.stroke_length_membership["short"],
            label="Short",
            color="blue"
        )
        axes[2].plot(
            self.attributes["stroke_length"],
            self.stroke_length_membership["medium"],
            label="Medium",
            color="green"
        )
        axes[2].plot(
            self.attributes["stroke_length"],
            self.stroke_length_membership["long"],
            label="Long",
            color="red"
        )
        axes[2].set_title("Stroke Length")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    def calculate_membership(self, curvature, angle, stroke_length):
        """Calculates membership values for given inputs."""
        curvature_membership = {
            k: fuzz.interp_membership(self.attributes["curvature"], v, curvature)
            for k, v in self.curvature_membership.items()
        }

        angle_membership = {
            k: fuzz.interp_membership(self.attributes["angle"], v, angle)
            for k, v in self.angle_membership.items()
        }

        stroke_length_membership = {
            k: fuzz.interp_membership(self.attributes["stroke_length"], v, stroke_length)
            for k, v in self.stroke_length_membership.items()
        }

        return {
            "curvature": curvature_membership,
            "angle": angle_membership,
            "stroke_length": stroke_length_membership,
        }

# Example Usage
if __name__ == "__main__":
    matcher = HandwrittenCharacterMatcher()
    matcher.visualize_memberships()

    # Example input features for a handwritten character
    input_curvature = 60
    input_angle = 120
    input_stroke_length = 30

    memberships = matcher.calculate_membership(
        input_curvature, input_angle, input_stroke_length
    )

    print("Membership values for curvature:", memberships["curvature"])
    print("Membership values for angle:", memberships["angle"])
    print("Membership values for stroke length:", memberships["stroke_length"])


"""output:
Membership values for curvature: {'low': 0.0, 'medium': 0.6, 'high': 0.2}
Membership values for angle: {'small': 0.0, 'medium': 0.3333333333333333, 'large': 0.3333333333333333}
Membership values for stroke length: {'short': 0.0, 'medium': 0.6666666666666666, 'long': 0.2}
"""


