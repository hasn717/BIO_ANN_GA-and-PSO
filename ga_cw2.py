# -*- coding: utf-8 -*-
"""GA_CW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15BR5HnHhOY6yztH6NabA18lZMIT3UhRj
"""

!pip install optproblems

import numpy as np
import matplotlib.pyplot as plt
from optproblems import cec2005
from optproblems import Individual
from datetime import datetime
from statistics import mean

class Solution:
  def __init__(self,individual,score=0):
    self.individual = individual
    self.score = score
  def __str__(self) -> str:
     return 'SCORE ' + str(self.score) +' | INDIVIDUAL'+ str(self.individual)

class Problem:
  def __init__(self,f):
    self.f = f
  
  def evaluate(self,sol):
      func =  self.f(len(sol))
      solution=Individual(sol)
      func.evaluate(solution)
      return solution.objective_values

  def optimal(self,length):
    func = self.f(length)
    solutions = func.get_optimal_solutions()
    for sol in solutions:
      func.evaluate(sol) # return single solution
      return sol.objective_values
    return 0

class GA:

  @staticmethod
  def one_point_crossover(p1, p2): ## cross over parents
    a1 = p1.individual[:]
    b1 = p2.individual[:]
    pick = np.random.randint(0,len(p1.individual)) 
    crossed1 = np.concatenate((b1[:pick],a1[pick:]))
    crossed2 = np.concatenate((a1[:pick],b1[pick:]))
    return crossed1,crossed2

  @staticmethod
  def tournament_selection(population,k=3): ## tournament selection 
    selected = np.random.randint(len(population))
    for index in np.random.randint(0, len(population), k-1):
      if population[index].score < population[selected].score:
        selected = index
    return population[selected]


  @staticmethod
  def single_gene_random_mutation(vector_length,individual,bounds): ## single gene mutation, picking random gen and giving it a random value
    if(len(bounds) != 2):
      raise Exception('Input range should be an array/tuple of length 2')
    rand_pick_gene = np.random.randint(0,vector_length)
    individual.individual[rand_pick_gene] = np.random.uniform(bounds[0],bounds[1])

  

  @staticmethod
  def fitness(objFunc,solution,best_solution): ## simple fitness function, picking the best score (lower is better)
    solution.score=objFunc.evaluate(solution.individual) 
    if solution.score < best_solution.score:
        best_solution = solution
    return best_solution

  
  @staticmethod
  def termination(objFunc,solution):
    if((objFunc.optimal(len(solution.individual)) + 5) > solution.score): ## termination state when optimal solution is reached with an error of 5
      return True
    return False


  '''
  GA algorithm that initializes a population, iterating over them while doing a fitness assessment, selection and breeding.
  -	Selection -> Selection function [tournament selection is used following coursework specification]
  -	Mutation -> Mutation function [single gene random mutation is used following coursework specification]
  -	Crossover -> Cross over function [one point crossover is used following coursework specification]
  -	Fitness -> Fitness function a simple conditional picking the best score (using evaluate method from Problem class) 
  -	Termination -> Termination function [Compare optimal solution with the given solution (with a small error added)]
  -	Objective function -> a function from the test problems (f1,f2,f3???)
  -	n_gen -> number of generations
  -	pop_size -> Length of the population 
  -	Bounds -> bounds of the solution (Capped depending on the test functions spec)
  -	Vector size -> The vector size of the solution
  -	Elite rate -> How much to keep from the parents 

  Elitism is used to inject fit parents to next generation (Exploitative Variation)
  '''
  @staticmethod
  def Evolve(objFunc,fitness,selection,termination,crossover,mutation,n_gen,pop_size,bounds,vector_size=10,elite_rate=0.5):
    if(len(bounds) != 2):
      raise Exception('Input range of bounds should be an array/tuple of length 2')
    if(pop_size % 2 != 0):## keep population size divisible by 2
      pop_size+=1 
    population=  [Solution(individual) for individual in np.random.uniform(low=bounds[0],high=bounds[1],size=(np.abs(pop_size),np.abs(vector_size)))] ## initialize population
    np.random.shuffle(population)
    best_solution = population[np.random.randint(0,len(population))]
    history = []
    for gen in range(n_gen):
      for solution in population: 
        best_solution = fitness(objFunc,solution,best_solution)
        history.append(best_solution.score)
        if termination(objFunc,best_solution):
          print('Terminated -> reached termination state')
          return best_solution,history
      selected_parents = [selection(population) for _ in range(len(population))] ## Selection
      children = []
      for mate in range(0, len(population), 2):
        p1, p2 = selected_parents[mate], selected_parents[mate+1]
        reproduce = crossover(p1,p2) # cross over
        for gene in reproduce:
          child= Solution(gene)
          mutation(vector_size,child,bounds) ## mutation
          children.append(child)

      children.sort(key=lambda individual: individual.score, reverse=False) ## sort fittest children by score
      population.sort(key=lambda individual: individual.score, reverse=False) ## sort fittest parents by score
      elites_rate = int(len(population)* np.abs(elite_rate))  ## Elitism, keep percentage of top parents for next generation  
      population[-elites_rate:] = children[:elites_rate if elites_rate > 0 else len(children)]
    return best_solution,history

"""# Evaluation & Experiments"""

f1 = Problem(cec2005.F1)
f2 = Problem(cec2005.F2)
f3 = Problem(cec2005.F3)
f4 = Problem(cec2005.F4)
f5 = Problem(cec2005.F5)
f6 = Problem(cec2005.F6)
f8 = Problem(cec2005.F8)

def run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,runs=50):
  ## Evaluation 
  evaluation = {
      'accuracies': [],
      'timedRuns': [],
      'runs': [],
      'pop_length': POPULATION_LENGTH,
      'n_gen': GENERATION_ITER
  }
  for run in range(runs):
    print('RUN ',run+1)
    start_time = datetime.now()
    solution,data = GA.Evolve(objFunc=OBJECTIVE_FUNC,
                        fitness=GA.fitness,
                        selection=GA.tournament_selection,
                        crossover=GA.one_point_crossover,
                        mutation=GA.single_gene_random_mutation,
                        n_gen=GENERATION_ITER,
                        pop_size=POPULATION_LENGTH,
                        vector_size=VECTOR_LENGTH,
                        termination=GA.termination,
                        bounds=BOUNDS,
                        elite_rate=ELITE_RATE)
    delta_time = datetime.now() - start_time 
    timeTaken = delta_time.total_seconds() * 1000
    accuracy = (solution.score / OBJECTIVE_FUNC.optimal(VECTOR_LENGTH)) * 100

    evaluation['timedRuns'].append(timeTaken)
    evaluation['accuracies'].append(accuracy)
    evaluation['runs'].append(data)


    print('Time taken->',timeTaken, 'ms')
    print('Fittest individual->',solution)
    print('Accuracy ->',accuracy,'%')
    print('#####################################')
  return evaluation

## Experiments parameetrs 
experiment_runs = 25

"""## Experiment 1

### F1
"""

## Running Experiment F1
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 1000
POPULATION_LENGTH = 10
BOUNDS = [-100,100]
ELITE_RATE = 0.5
OBJECTIVE_FUNC=f1
exp_f1 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""### F4"""

## Running Experiment F4
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 1500
POPULATION_LENGTH = 100
BOUNDS = [-100,100]
ELITE_RATE = 0.5
OBJECTIVE_FUNC=f4
exp_f4 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""### F8"""

## Running Experiment F8
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 100
POPULATION_LENGTH = 6
BOUNDS = [-32,32]
ELITE_RATE = 0.5
OBJECTIVE_FUNC = f8
exp_f8 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""## Analysis

### F1, F4 ,F8
"""

x_axis = [i for i in range(experiment_runs)]
plt.plot(x_axis, exp_f1['accuracies'])
plt.plot(x_axis, exp_f4['accuracies'])
plt.plot(x_axis, exp_f8['accuracies'])

plt.title('Accuracy vs Runs')
plt.xlabel('Runs')
plt.ylabel('Accuracy(%)')
plt.legend(['F1', 'F4','F8'])
plt.show()

"""### F1"""

## F1 Convergence Single Run
x_axis = [i for i in range(len(exp_f1['runs'][0]))]
plt.plot(x_axis, exp_f1['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

"""### F4"""

## F4 Convergence Single Run
x_axis = [i for i in range(len(exp_f4['runs'][0]))]
plt.plot(x_axis, exp_f4['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

"""### F8"""

## F8 Convergence Single Run
x_axis = [i for i in range(len(exp_f8['runs'][0]))]
plt.plot(x_axis, exp_f8['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

best_exp_f1 = np.max(exp_f1['accuracies'])
print('Best F1 Accuracy->',best_exp_f1)
best_exp_f4 = np.max(exp_f4['accuracies'])
print('Best F4 Accuracy->',best_exp_f4)
best_exp_f8 = np.max(exp_f8['accuracies'])
print('Best F8 Accuracy->',best_exp_f8)

worst_exp_f1 = np.min(exp_f1['accuracies'])
print('Worst F1 Accuracy->',worst_exp_f1)
worst_exp_f4 = np.min(exp_f4['accuracies'])
print('Worst F4 Accuracy->',worst_exp_f4)
worst_exp_f8 = np.min(exp_f8['accuracies'])
print('Worst F8 Accuracy->',worst_exp_f8)

median_exp_f1 = np.median(exp_f1['accuracies'])
print('median F1 Accuracy->',median_exp_f1)
median_exp_f4 = np.median(exp_f4['accuracies'])
print('median F4 Accuracy->',median_exp_f4)
median_exp_f8 = np.median(exp_f8['accuracies'])
print('median F8 Accuracy->',median_exp_f8)

reliability_exp_f1 = mean(exp_f1['accuracies'])
accuracy_exp_f1 = mean(exp_f1['accuracies'][0:20])
efficiency_exp_f1 = mean(exp_f1['timedRuns'])

print("Reliability",reliability_exp_f1)
print("Accuracy",accuracy_exp_f1)
print("Efficiency",efficiency_exp_f1)

reliability_exp_f4 = mean(exp_f4['accuracies'])
accuracy_exp_f4 = mean(exp_f4['accuracies'][0:20])
efficiency_exp_f4 = mean(exp_f4['timedRuns'])

print("Reliability",reliability_exp_f4)
print("Accuracy",accuracy_exp_f4)
print("Efficiency",efficiency_exp_f4)

reliability_exp_f8 = mean(exp_f8['accuracies'])
accuracy_exp_f8 = mean(exp_f8['accuracies'][0:20])
efficiency_exp_f8 = mean(exp_f8['timedRuns'])

print("Reliability",reliability_exp_f8)
print("Accuracy",accuracy_exp_f8)
print("Efficiency",efficiency_exp_f8)

"""## Experiment 2

### F1
"""

## Running Experiment F1
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 250
POPULATION_LENGTH = 10
BOUNDS = [-100,100]
ELITE_RATE = 0.5
OBJECTIVE_FUNC=f1
exp2_f1 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""### F4"""

## Running Experiment F4
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 250
POPULATION_LENGTH = 10
BOUNDS = [-100,100]
ELITE_RATE = 0.5
OBJECTIVE_FUNC=f4
exp2_f4 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""### F8"""

## Running Experiment F8
## Hyperparameters
VECTOR_LENGTH = 10
GENERATION_ITER = 250
POPULATION_LENGTH = 6
BOUNDS = [-32,32]
ELITE_RATE = 0.5
OBJECTIVE_FUNC = f8
exp2_f8 = run_experiment(OBJECTIVE_FUNC,VECTOR_LENGTH,GENERATION_ITER,POPULATION_LENGTH,BOUNDS,ELITE_RATE,experiment_runs)

"""## Analysis

### F1, F4 ,F8
"""

x_axis = [i for i in range(experiment_runs)]
plt.plot(x_axis, exp2_f1['accuracies'])
plt.plot(x_axis, exp2_f4['accuracies'])
plt.plot(x_axis, exp2_f8['accuracies'])

plt.title('Accuracy vs Runs')
plt.xlabel('Runs')
plt.ylabel('Accuracy(%)')
plt.legend(['F1', 'F4','F8'])
plt.show()

"""### F1"""

## F1 Convergence Single Run
x_axis = [i for i in range(len(exp2_f1['runs'][0]))]
plt.plot(x_axis, exp2_f1['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

"""### F4"""

## F4 Convergence Single Run
x_axis = [i for i in range(len(exp2_f4['runs'][0]))]
plt.plot(x_axis, exp2_f4['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

"""### F8"""

## F8 Convergence Single Run
x_axis = [i for i in range(len(exp2_f8['runs'][0]))]
plt.plot(x_axis, exp2_f8['runs'][0])
plt.title('Convergence')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

best_exp2_f1 = np.max(exp2_f1['accuracies'])
print('Best F1 Accuracy->',best_exp2_f1)
best_exp2_f4 = np.max(exp2_f4['accuracies'])
print('Best F4 Accuracy->',best_exp2_f4)
best_exp2_f8 = np.max(exp2_f8['accuracies'])
print('Best F8 Accuracy->',best_exp2_f8)

worst_exp2_f1 = np.min(exp2_f1['accuracies'])
print('Worst F1 Accuracy->',worst_exp2_f1)
worst_exp2_f4 = np.min(exp2_f4['accuracies'])
print('Worst F4 Accuracy->',worst_exp2_f4)
worst_exp2_f8 = np.min(exp2_f8['accuracies'])
print('Worst F8 Accuracy->',worst_exp2_f8)

median_exp2_f1 = np.median(exp2_f1['accuracies'])
print('median F1 Accuracy->',median_exp2_f1)
median_exp2_f4 = np.median(exp2_f4['accuracies'])
print('median F4 Accuracy->',median_exp2_f4)
median_exp2_f8 = np.median(exp2_f8['accuracies'])
print('median F8 Accuracy->',median_exp2_f8)

reliability_exp2_f1 = mean(exp2_f1['accuracies'])
accuracy_exp2_f1 = mean(exp2_f1['accuracies'][0:20])
efficiency_exp2_f1 = mean(exp2_f1['timedRuns'])

print("Reliability",reliability_exp2_f1)
print("Accuracy",accuracy_exp2_f1)
print("Efficiency",efficiency_exp2_f1)

reliability_exp2_f4 = mean(exp2_f4['accuracies'])
accuracy_exp2_f4 = mean(exp2_f4['accuracies'][0:20])
efficiency_exp2_f4 = mean(exp2_f4['timedRuns'])

print("Reliability",reliability_exp2_f4)
print("Accuracy",accuracy_exp2_f4)
print("Efficiency",efficiency_exp2_f4)

reliability_exp2_f8 = mean(exp2_f8['accuracies'])
accuracy_exp2_f8 = mean(exp2_f8['accuracies'][0:20])
efficiency_exp2_f8 = mean(exp2_f8['timedRuns'])

print("Reliability",reliability_exp2_f8)
print("Accuracy",accuracy_exp2_f8)
print("Efficiency",efficiency_exp2_f8)