# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
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
        newFood = successorGameState.getFood()
        height, width = newFood.height, newFood.width

        succ_capsulesLocations = successorGameState.getCapsules()
        succ_numCapsulesLeft = len(succ_capsulesLocations)
        curr_capsulesLocations = currentGameState.getCapsules()
        curr_numCapsulesLeft = len(curr_capsulesLocations)
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostsScared = [ghostState.scaredTimer != 0 for ghostState in newGhostStates]
        nextScore = successorGameState.getScore()

        #calculate distance to ghosts
        dist_to_ghosts = [util.manhattanDistance( newPos, ghostPos) for ghostPos in newGhostPositions]

        #calculate ghost reflax
        chase_ghosts_value = [ch_val[0]-ch_val[1] for ch_val in zip(newScaredTimes,dist_to_ghosts)]
        chase_ghosts_score = sum([ch_scr[0]*ch_scr[1] for ch_scr in zip(ghostsScared,chase_ghosts_value)])
        flee_ghosts_score = sum([(not fl_scr[0])*fl_scr[1] for fl_scr in zip(ghostsScared,dist_to_ghosts)])
        ghost_reflax = -20/(1+flee_ghosts_score)
        #was a dot eaten?
        ate_a_dot = currentGameState.getNumFood() - successorGameState.getNumFood()

        #calculate minimum and average manhattan distance to food
        dist_to_closest_food = height + width + 1
        for i in range(width):
            for j in range(height):
                if newFood[i][j]:
                    dist = util.manhattanDistance( newPos, (i,j) )
                    if  dist < dist_to_closest_food:
                        dist_to_closest_food = dist

        if dist_to_closest_food == height + width + 1:
            dist_to_closest_food = 0


        "*** YOUR CODE HERE ***"
        evaluation_score =  successorGameState.getScore() + (height + width + 1)*ate_a_dot - 2*dist_to_closest_food + ghost_reflax# + 50*ate_a_capsule + capsule_reflax
        return  evaluation_score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        v, action = self.maxValue(gameState, 0)
        #print(v, action, self.depth)
        return action

    def maxValue(self, state, prev_depth):
        depth = prev_depth+1
        if depth > self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        best_v, best_action = -float("inf"), None
        pacman_idx, agent_start_idx = 0, 1
        """
        print(depth, "pacman has " + str(len(state.getLegalActions(pacman_idx)))\
                +"legal actions")
                """
        for action in state.getLegalActions(pacman_idx):
            successor_state = state.generateSuccessor(pacman_idx, action)
            v,_ = self.minValue(successor_state, depth, agent_start_idx)
            if v > best_v:
                best_v, best_action = v, action
        return best_v, best_action

    def minValue(self, state, depth, agent_idx):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        # func(state, depth) 
        if agent_idx == state.getNumAgents() - 1:
            func = self.maxValue
        else:
            next_agent_idx = agent_idx+1
            func = lambda nstate, depth: self.minValue(nstate, depth,
                    next_agent_idx)

        best_v, best_action = float("inf"), None
        """
        print(depth, "agent"+str(agent_idx)+" has " \
                + str(len(state.getLegalActions(agent_idx)))\
                +"legal actions")
                """
        for action in state.getLegalActions(agent_idx):
            successor_state = state.generateSuccessor(agent_idx, action)
            v,_ = func(successor_state, depth)
            if v < best_v:
                best_v, best_action = v, action
        return best_v, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
