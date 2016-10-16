# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            values = self.values.copy()
            for state in self.mdp.getStates():
                action_value = []
                if not self.mdp.isTerminal(state):                
                    for action in self.mdp.getPossibleActions(state):
                        action_value.append(self.computeQValueFromValues(state, action))
                    values[state] = max(action_value)
            self.values = values
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        t = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0
        for (nextState, p) in t:
            q += p * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        values = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            values[action]=self.getQValue(state, action)
            
        return values.argMax()

            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, action)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = {}
        priorityQueue = util.PriorityQueue()
        for s in states:
            predecessors[s] = set()
            
        for s in states:
            for action in self.mdp.getPossibleActions(s):
                t = self.mdp.getTransitionStatesAndProbs(s, action)              
                for i in t:
                    if i[1] > 0:
                        predecessors[i[0]].add(s)

        for s in states:
            if not self.mdp.isTerminal(s):
                qvalue = []
                for action in self.mdp.getPossibleActions(s):
                    qvalue.append(self.getQValue(s, action))
                diff = abs(self.values[s] - max(qvalue))
                priorityQueue.push(s, -diff) 
                
        for i in range(self.iterations):
            if not priorityQueue.isEmpty():
                s = priorityQueue.pop()
                qvalue1 = []
                for action in self.mdp.getPossibleActions(s):
                    qvalue1.append(self.getQValue(s, action))
                self.values[s] = max(qvalue1)
                
                for p in predecessors[s]:
                    qvalue = []
                    for action in self.mdp.getPossibleActions(p):
                        qvalue.append(self.getQValue(p, action))
                    diff = abs(self.values[p] - max(qvalue))

                    if diff > self.theta:
                        priorityQueue.update(p, -diff)
            else:
                break
                    
        
        
        
        
       

