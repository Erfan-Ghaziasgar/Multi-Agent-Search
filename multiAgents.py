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


from math import inf

import util
from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


def betterEvaluationFunction(game_state: GameState):

    WIN_FACTOR = 5000
    LOST_FACTOR = -50000
    FOOD_COUNT_FACTOR = 1_000_000
    FOOD_DISTANCE_FACTOR = 1_000
    CAPSULES_FACTOR = 10_000

    food_distance = distance_to_nearest(
        game_state, game_state.getFood().asList())
    ghost_distance = distance_to_nearest(
        game_state, game_state.getGhostPositions())
    food_count = game_state.getNumFood()
    capsules_count = len(game_state.getCapsules())

    food_count_value = 1 / (food_count + 1) * FOOD_COUNT_FACTOR
    ghost_value = ghost_distance
    food_distance_value = 1 / food_distance * FOOD_DISTANCE_FACTOR
    capsules_count_value = 1 / (capsules_count + 1) * CAPSULES_FACTOR
    end_value = 0

    if game_state.isLose():
        end_value += LOST_FACTOR
    elif game_state.isWin():
        end_value += WIN_FACTOR

    return food_count_value + ghost_value + food_distance_value + \
        capsules_count_value + end_value


def distance_to_nearest(game_state, positions):
    if len(positions) == 0:
        return inf
    pacman_pos = game_state.getPacmanPosition()
    distances = [util.manhattanDistance(pacman_pos, position)
                 for position in positions]
    return min(distances)


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="betterEvaluationFunction", depth="2",
                 time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Here are some method calls that might be useful when implementing
        minimax.

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

        action, _ = self.min_max_value(
            game_state=gameState,
            agent_index=0,
            alpha=-inf,
            beta=inf,
            depth=0
        )
        return action

    def min_max_value(self, game_state: GameState, agent_index,
                      alpha, beta, depth):
        if game_state.isWin() or game_state.isLose() or depth >= \
                self.depth * game_state.getNumAgents():
            return 'Stop', self.evaluationFunction(game_state)
        elif agent_index == 0:
            return self.max_value(game_state, 0, alpha, beta, depth)
        else:
            return self.min_value(game_state, agent_index, alpha, beta, depth)

    def max_value(self, game_state: GameState, agent_index, alpha,
                  beta, depth):
        best_value = -inf
        best_action = 'Stop'
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()

        for next_action in game_state.getLegalActions(agent_index):
            next_state = game_state.generateSuccessor(agent_index, next_action)
            next_value = self.min_max_value(
                next_state, next_agent_index, alpha, beta, depth + 1)[1]
            if next_value > best_value:
                best_action, best_value = next_action, next_value
            if best_value > beta:
                return best_action, best_value
            alpha = max(alpha, best_value)
        return best_action, best_value

    def min_value(self, game_state: GameState, agent_index, alpha,
                  beta, depth):
        best_value = inf
        best_action = 'Stop'
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()

        for next_action in game_state.getLegalActions(agent_index):
            next_state = game_state.generateSuccessor(agent_index, next_action)
            next_value = self.min_max_value(
                next_state, next_agent_index, alpha, beta, depth + 1)[1]
            if next_value < best_value:
                best_action, best_value = next_action, next_value
            if best_value < alpha:
                return best_action, best_value
            beta = min(beta, best_value)
        return best_action, best_value
