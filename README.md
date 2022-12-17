# pacman #13

## Overview

This project involves programming agents for the classic version of Pacman, including ghosts, implementing Minimax and Expectimax searches, and creating evaluation functions. The goal is to create intelligent agents that can play Pacman effectively, using various search algorithms and evaluation techniques.


## Run

To play Pacman, use the following command in the command line:
`python pacman.py`

Pacman supports several options. To see a list of all options and their default values, use:
`python pacman.py -h`


## q1

### a reflex agent consider both food locations and ghost locations

* watch:
  `python pacman.py -p ReflexAgent -l testClassic`
  `python pacman.py --frameTime 0 -p ReflexAgent -k 1`
  `python pacman.py --frameTime 0 -p ReflexAgent -k 2`

* test: `python autograder.py -q q1`
  `python autograder.py -q q1 --no-graphics`

The reflex agent's evaluation function helps Pacman make decisions about which actions to take in the game by evaluating the potential benefits of each action. It begins by ensuring that Pacman avoids colliding with ghosts, as this would result in a loss. After avoiding ghosts, Pacman calculates the distance to the closest food and returns the current score and the value of the nearest food. This enables Pacman to prioritize actions that are likely to result in the most points or the greatest benefit.


## q2

### Minimax agents can work with any number of ghosts

* watch:
  `python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4`
  `python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3`

* test: `python autograder.py -q q2`
  `python autograder.py -q q2 --no-graphics`

The Minimax agent uses the get_minimax_val function to calculate Pacman's best action based on the minmax value. If the search reaches a maximum depth or an illegal move, the get_minimax_val function returns zero. If the index is zero, it's Pacman's turn to maximize the value. Otherwise, it's a ghost's turn to minimize it. These two functions use a recursive approach to search through all possible actions and states, and return the value and the best action for the current state. The value alternates between maximizing and minimizing depending on whether Pacman is playing or a ghost is.


## q3

### A minimax tree exploration agent that uses alpha beta pruning

* watch:
  `python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic`

* test: `python autograder.py -q q3`
  `python autograder.py -q q3 --no-graphics`

In order to create the Alpha Beta Agent, the Minimax algorithm was used and the following changes were made:

Before starting the search, beta and alpha were initialized to the largest positive and negative numbers respectively.
The get_minimax_val function was updated to include alpha and beta parameters.
The max_value function was updated to include an alpha-beta pruning check, returning the value of alpha as the maximum value if alpha is greater than beta.
The min_value function was updated to include an alpha-beta pruning check, returning the value of beta as the minimum value if beta is smaller than alpha.
As a result of these changes, the Minimax algorithm can now incorporate alpha-beta pruning, which can significantly improve its efficiency by pruning non-relevant branches from the search tree.


## q4

### Using the ExpectimaxAgent, suboptimal agents can be modeled probabilistically.

* watch:
  `python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3`
  `python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10`
  `python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10`

* test: `python autograder.py -q q4`
  `python autograder.py -q q4 --no-graphics`

The Expectimax Agent uses a recursive search algorithm similar to the Minimax Agent, but with some key differences. Rather than alternating between maximizing and minimizing at each level of the search tree, the Expectimax Agent estimates the expected value of each action based on the probabilities of different outcomes. This allows the agent to make decisions that are not necessarily optimal, but are still reasonable given the uncertainty of the game.


## q5

* test: `python autograder.py -q q5`
  `python autograder.py -q q5 --no-graphics`

### a better evaluation function for Pacman
The evaluation function returns a score that reflects how good or bad the current state is for the Pacman player.
The score is calculated by considering several factors:
Closest food: The distance to the closest food pellet is calculated using the manhattanDistance function, which returns the Manhattan distance. A smaller distance means that the Pacman is closer to the food, so a higher score is returned.
Food left: The number of food pellets remaining in the game is multiplied by a large constant to give it more weight in the final score.
Capsules left: The number of power capsules remaining in the game is multiplied by a smaller constant to give it less weight in the final score.
Distance to ghost: The distance to the nearest ghost is calculated and added to the score. A larger distance means a higher score. If a ghost is too close (within 2 units), the function returns negative infinity to indicate that the current state is a losing state.
Additional factors: If the current state is a losing state (Pacman has been eaten by a ghost), a large negative constant is subtracted from the score. If the current state is a winning state (all food has been eaten), a large positive constant is added to the score.
Finally, the score is returned. 
By incorporating these and other factors into the evaluation function, we can create a more effective agent that is better able to make strategic decisions in the game.

