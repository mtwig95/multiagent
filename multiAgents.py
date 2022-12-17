# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        # if the ghost close to pacman then go
        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghost) < 2:
                return -float('inf')

        # Find the nearest food dot
        closest_food = float("inf")
        for food in newFood:
            closest_food = min(closest_food, manhattanDistance(newPos, food))

        # return the scoring and the closest food location
        return 1.0 / closest_food + successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        result = self.get_minimax_val(gameState, 0, 0)
        return result[1]

    def get_minimax_val(self, gameState, index, depth):
        # Check if there isn't legal move maximum depth has been reached, Return the value of the terminal state
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # Pac-Man's turn: maximize the value
        if index == 0:
            return self.max_value(gameState, index, depth)

        # Ghost's turn: minimize the value
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        max_value = float("-inf")

        # Get the list of legal actions for the current agent
        for action in gameState.getLegalActions(index):
            # generate the next state after taking the given action
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            optional_value = self.get_minimax_val(next_state, next_index, next_depth)[0]

            # Update the maximum value and best action if necessary
            if optional_value > max_value:
                max_value = optional_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
        min_value = float("inf")

        # Get the list of legal actions for the current agent
        for action in gameState.getLegalActions(index):
            # generate the next state after taking the given action
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            optional_value = self.get_minimax_val(next_state, next_index, next_depth)[0]

            # Update the minimum value and best action if necessary
            if optional_value < min_value:
                min_value = optional_value
                min_action = action

        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha_beta_val = self.get_minimax_val(gameState, 0, 0, float("-inf"), float("inf"))
        return alpha_beta_val[1]

    def get_minimax_val(self, game_state, index, depth, alpha, beta):
        # Check if there isn't legal move maximum depth has been reached, Return the value of the terminal state
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return game_state.getScore(), ""

        # Pac-Man's turn: maximize the value
        if index == 0:
            return self.max_value(game_state, index, depth, alpha, beta)

        # Ghost's turn: minimize the value
        else:
            return self.min_value(game_state, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        max_value = float("-inf")

        # Get the list of legal actions for the current agent
        for action in gameState.getLegalActions(index):
            # generate the next state after taking the given action
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            optional_value, optional_action = self.get_minimax_val(next_state, next_index, next_depth, alpha, beta)

            # Update the maximum value and best action if necessary
            if optional_value > max_value:
                max_value = optional_value
                max_action = action

            alpha = max(alpha, max_value)

            if max_value > beta:
                return max_value, max_action

        return max_value, max_action

    def min_value(self, gameState, index, depth, alpha, beta):
        min_value = float("inf")

        # Get the list of legal actions for the current agent
        for action in gameState.getLegalActions(index):
            # generate the next state after taking the given action
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            optional_value, optional_action = self.get_minimax_val(next_state, next_index, next_depth, alpha, beta)

            # Update the minimum value and best action if necessary
            if optional_value < min_value:
                min_value = optional_value
                min_action = action

            beta = min(beta, min_value)

            if min_value < alpha:
                return min_value, min_action

        return min_value, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.get_value(gameState, 0, 0)  ##### 1

        return action

    def get_value(self, gameState, index, depth):
        # Check if there isn't legal move maximum depth has been reached, Return the value of the evaluation function
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(gameState)

        # Pac-Man's turn: maximize the value
        if index == 0:
            return self.max_value(gameState, index, depth)

        # Ghost's turn: minimize the value
        else:
            return self.expected_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        max_value = float("-inf")
        max_action = ""

        # Get the list of legal actions for the current agent
        for action in gameState.getLegalActions(index):
            # generate the next state after taking the given action
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            optional_action, optional_value = self.get_value(next_state, next_index, next_depth)

            # Update the maximum value and best action if necessary
            if optional_value > max_value:
                max_value = optional_value
                max_action = action

        return max_action, max_value

    def expected_value(self, gameState, index, depth):
        # Initialize the expected value to zero
        expected_value = 0

        # Iterate through all the possible actions
        for action in gameState.getLegalActions(index):
            next_state = gameState.generateSuccessor(index, action)
            next_index = index + 1
            next_depth = depth

            # Update the next_state agent's index and depth if it's pacman
            if next_index == gameState.getNumAgents():
                next_index = 0
                next_depth += 1

            # Calculate the value of the next state
            next_action, next_val = self.get_value(next_state, next_index, next_depth)

            # Add the probability-weighted value to the expected value
            expected_value += (1.0 / len(gameState.getLegalActions(index))) * next_val

        return next_action, expected_value


def betterEvaluationFunction(currentGameState):
    # Get the current position of Pac-Man and the list of food pellets
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()

    # Calculate the minimum distance to a food pellet
    if food_list:
        min_food_distance = min(manhattanDistance(pacman_pos, food) for food in food_list)
    else:
        # If there are no food pellets remaining, set the minimum distance to a large value
        min_food_distance = float('inf')

    # Calculate the minimum distance to a ghost
    ghost_positions = currentGameState.getGhostPositions()
    if ghost_positions:
        min_ghost_distance = min(manhattanDistance(pacman_pos, ghost) for ghost in ghost_positions)
    else:
        # If there are no ghosts remaining, set the minimum distance to a large value
        min_ghost_distance = float('inf')

    # Get the number of food pellets and power capsules remaining in the game
    food_left = currentGameState.getNumFood()
    capsule_left = len(currentGameState.getCapsules())

    # Calculate the score using the factors described above
    score = (1.0 / (food_left + 1)) * 1000000 + min_ghost_distance + (1.0 / (min_food_distance + 1)) * 1000 + (
            1.0 / (capsule_left + 1)) * 10000
    if currentGameState.isLose():
        score -= 50000
    elif currentGameState.isWin():
        score += 50000

    return score


# Abbreviation
better = betterEvaluationFunction
