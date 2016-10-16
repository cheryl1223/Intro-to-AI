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
        chosenIndex = 0
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
        newPos = list(successorGameState.getPacmanPosition())
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        foodList = currentGameState.getFood().asList()
        foodDist = []
        pacmanPos = list(successorGameState.getPacmanPosition())
        score = 0

        for i in range(len(newGhostStates)):
            ghost_dist = util.manhattanDistance(pacmanPos, newGhostStates[i].getPosition())
            if ghost_dist < 3 and ghostState.scaredTimer is 0:
                score = -float("inf")
                return score
        for food in foodList:
            x = -abs(food[0] - pacmanPos[0])
            y = -abs(food[1] - pacmanPos[1])
            foodDist.append(x+y) 
            
        capsules = currentGameState.getCapsules()
        capsuleDist = []
        nearest = 0
        for capsule in capsules:
            x = -abs(capsule[0] - pacmanPos[0])
            y = -abs(capsule[1] - pacmanPos[1])
            capsuleDist.append(x+y) 
        if len(capsuleDist)>0:
            nearest = max(capsuleDist)
        score = max(foodDist) + nearest 
        return score
       

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState,0)[0]

    def minimax(self, gameState, depth):
        agent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        
        if agent == 0:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            max_value = (None,-float("inf"))
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.minimax(child,depth+1)
                if v[1] > max_value[1]:
                    max_value = (action, v[1])
            return max_value
        else:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            min_value = (None,float("inf"))
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.minimax(child,depth+1)
                if v[1] < min_value[1]:
                    min_value = (action, v[1])
            return min_value
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(gameState,0,-float("inf"), float("inf"))[0]

    def alphaBeta(self, gameState, depth, alpha, beta):
        agent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))

        if agent == 0:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            max_value = (None,-float("inf"))
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.alphaBeta(child,depth+1,alpha,beta)
                if v[1]>max_value[1]:
                    max_value = (action, v[1])
                if max_value[1] > beta:
                    return max_value
                alpha = max(alpha,max_value[1])
            return max_value

        else:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            min_value = (None,float("inf"))
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.alphaBeta(child,depth+1,alpha,beta)
                if v[1]<min_value[1]:
                    min_value = (action, v[1])
                if min_value[1] < alpha:
                    return min_value
                beta = min(beta,min_value[1])       
            return min_value
            
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
        return self.expectiMax(gameState,0)[0]

    def expectiMax(self, gameState, depth):
        agent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))

        if agent == 0:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            max_value = (None,-float("inf"))
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.expectiMax(child,depth+1)
                if v[1] > max_value[1]:
                    max_value = (action, v[1])
            return max_value

        else:
            if len(actions) == 0:
                return (None,self.evaluationFunction(gameState))
            exp_value = (None,0)
            p = 1./len(actions)
            for action in actions:
                child = gameState.generateSuccessor(agent, action)
                v = self.expectiMax(child,depth+1)[1]*p
                exp_value = (action,exp_value[1]+v)
            return exp_value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()

    for i in range(len(newGhostStates)):
        ghost_dist = util.manhattanDistance(newPos, newGhostStates[i].getPosition())
        if ghost_dist < 3:
            score += -1000

    food = newFood.asList()
    foodDist = map(lambda xy: util.manhattanDistance(xy, newPos), food)
    for i in foodDist:
        score += 10./i
    score -= len(food)*5

    capsules = currentGameState.getCapsules()
    capsuleDist = map(lambda xy: util.manhattanDistance(xy, newPos), capsules)
    if len(capsuleDist)>0:
        nearest = min(capsuleDist)
    s_capsule = -8 if all((t == 0 for t in newScaredTimes)) else 0
    score += len(capsules) *s_capsule

    return score

# Abbreviation
better = betterEvaluationFunction

