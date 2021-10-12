"""
This file contains the code for group 33 for the assignment "EVOLUTIONARY COMPUTING:
STANDARD ASSIGNMENT - TASK I" 

An evolutionary algorithm is used to train a neural network to play evoman, which
is a python framework.

This is the random version of our implementation of an EA.
"""
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import random
import numpy as np
from math import fabs,sqrt
import glob, os
from random import randrange
from numpy.random import choice
import copy
np.set_printoptions(threshold = 10)
n_hidden_neurons = 10

# Creating a file with the name of experiment.
experiment_name = 'random_8'

if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[7,8],
                  multiplemode="yes",
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
				  enemymode="static",
				  level=2,
				  speed="fastest",
				  randomini='yes')

env.state_to_log()

# Number of nodes in the neural network
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Parameters 
pop_size = 100
gens = 20
mutation_chance = 0.4
gens_no_change = 5
mutation_range = 0.4
diversify_range = 0.4


def my_cons_multi(values):
	return values.mean()
 
 
env.cons_multi = my_cons_multi


# Plays the game with the given genes, returns list of fitness of all genes
def evaluate(genes):
	fit_list = []
	
	for gene in genes:
		_, Plife, Elife, _ = env.play(pcont=gene)
		fit_list.append(fitness(env, Plife, Elife))
	return fit_list
	
# Tests the fitness of the genes by checking player health and enemy health, doesn't go below 0
def fitness(env, Plife, Elife):
	if env.get_time() == 1000:
		return 1
	return 101 - Elife + Plife

# Mates best genes with eachother to create offspring, takes a random portion of both genes
def create_offspring(bg, bf):
	offspring = []
	shuffled = list(zip(bg, bf))
	random.shuffle(shuffled)
	bg, bf = zip(*shuffled)

	for i in range(0,len(bg),1):
		if i == len(bg) - 1:
			offspring.append((bg[i] * random.uniform(0, 1)) + (bg[0] * random.uniform(0, 1)))
			continue
		offspring.append((bg[i] * random.uniform(0, 1)) + (bg[i+1] * random.uniform(0, 1)))
		
		for i in range(len(offspring)):
			if mutation_chance > random.uniform(0, 1):
				for j in range(len(offspring[i])):
					offspring[i][j] += random.uniform(-mutation_range, mutation_range)
	return offspring

# Compares genes to eachother selects the highest one
def select_genes(fitness_pops, pop_genes):
	best_genes = []
	best_fitness = []
	
	for i in range(0,len(fitness_pops),1):
		if i == len(fitness_pops) - 1:
			if fitness_pops[i] > fitness_pops[0]:
				best_genes.append(pop_genes[i])
				best_fitness.append(fitness_pops[i])
			else:
				best_fitness.append(fitness_pops[0])
				best_genes.append(pop_genes[0])
			continue
			
		if fitness_pops[i] > fitness_pops[i+1]:
			best_fitness.append(fitness_pops[i])
			best_genes.append(pop_genes[i])
		else:
			best_fitness.append(fitness_pops[i+1])
			best_genes.append(pop_genes[i+1])
			
	return np.asarray(best_genes), np.asarray(best_fitness)

# Re-initialises half of the genes when no solution is reached
def diversify_deluxe(genes):
	for i in range(1, len(genes), 2):
		for j in range(len(genes[i])):
			genes[i][j] = random.uniform(-1, 1)
	return genes	

# Add some diversity to the genes
def diversify(genes):
	for i in range(0, len(genes), 2):
		for j in range(len(genes[i])):
			genes[i][j] = genes[i][j] + random.uniform(-diversify_range, diversify_range)
	return genes

# Starting the algorithm
if __name__ == '__main__':
	
	# run experiment X amount of times
	for run in range(1):
		try:
			os.makedirs(experiment_name + "/" + str(run+1))
		except:
			pass
		ini = time.time()	  
		print('\n STARTING EVOLUTION \n')
		
		# Initialazing the population
		pops = np.random.uniform(-1, 1, (pop_size, n_vars))

		
		# running the environment
		fitness_pops = evaluate(pops)
		
		# Sorting genes by fitness
		best_genes, best_fitness = select_genes(fitness_pops, pops)
		best_index = np.argmax(best_fitness)
		best_pop = (best_fitness[best_index], best_genes[best_index])

		
		# Creating offspring with best parents.
		offspring_genes = create_offspring(best_genes, best_fitness)
		fitness_offspring = evaluate(offspring_genes)
		
		total_new_genes = list(offspring_genes) + list(pops)
		total_new_fitness = fitness_offspring + fitness_pops
		
		best_index = np.argmax(total_new_fitness)
		new_best_pop = (total_new_fitness[best_index], total_new_genes[best_index])
	
		if new_best_pop[0] > best_pop[0]:
			best_pop = new_best_pop
		
		print(best_pop[0])
		
		mean = np.mean(total_new_fitness)
		std = np.std(total_new_fitness)
		total_new_fitness = np.asarray(total_new_fitness)
		weights = total_new_fitness / sum(total_new_fitness)
		
		
		# Creates a weighted average where higher fitnesses have a higher chance of surviving
		survivors_ints = choice(pop_size * 2, pop_size , p=weights, replace=False)
		survivor_genes = [total_new_genes[i] for i in survivors_ints]
		survivor_fitness = [total_new_fitness[i] for i in survivors_ints]
		
		survivor_genes[0] = best_pop[1]
		survivor_fitness[0] = best_pop[0]
		best_last = best_pop
		
		total_gen_no_change = 0
		
		# Creating a csv file for the results before entering loop.
		file_aux = open(experiment_name + "/" + str(run+1) + "/results.csv", "a")
		file_aux.write('generation;'+"best_fit;"+"mean;"+"std;"+'time'+"\n")
		file_aux.close()
		
		
		# Entering loop for evolution
		for generation in range(gens):

			print("New generation")
			#keeping track of generation while running
			print(generation+1)
			
			# new generation
			pops = survivor_genes
			fitness_pops = survivor_fitness
			
			best_genes, best_fitness = select_genes(fitness_pops, pops)
		
			best_index = np.argmax(best_fitness)
			new_best_pop = (best_fitness[best_index], best_genes[best_index])

			# check if best_pop has increased
			if new_best_pop[0] > best_pop[0]:
				best_pop = new_best_pop
		
			print("best fit")
			print(best_pop[0])
			
			
			# Selecting best genes out the new generation.
			offspring_genes = create_offspring(best_genes, best_fitness)
			fitness_offspring = evaluate(offspring_genes)
		
			total_new_genes = list(offspring_genes) + list(pops)
			total_new_fitness = fitness_offspring + fitness_pops
		
			# save best_pop
			best_index = np.argmax(total_new_fitness)
			new_best_pop = (total_new_fitness[best_index], total_new_genes[best_index])
		
			if new_best_pop[0] > best_pop[0]:
				best_pop = new_best_pop
			
			mean = np.mean(total_new_fitness)
			std = np.std(total_new_fitness)
			total_new_fitness = np.asarray(total_new_fitness)
			
			# Compute weighted fitness list to give higher fitness a higher chance to survive
			weights = total_new_fitness / sum(total_new_fitness)
			survivors_ints = choice(pop_size * 2, pop_size , p=weights, replace=False)
			survivor_genes = [total_new_genes[i] for i in survivors_ints]
			survivor_fitness = [total_new_fitness[i] for i in survivors_ints]
		
			# Insert best pop in population
			survivor_genes[int(len(survivor_genes)/2)] = best_pop[1]
			survivor_fitness[int(len(survivor_genes)/2)] = best_pop[0]
			
			# Checking if no evolution is happening
			if best_pop[0] == best_last[0]:
				total_gen_no_change += 1
				best_pop1 = copy.deepcopy(best_pop)
				# Adding new genes to diversify
				print("Adding new genes@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				survivor_genes = diversify(survivor_genes)
				if total_gen_no_change % gens_no_change == 0:
					survivor_genes = diversify_deluxe(survivor_genes)
				survivor_fitness = evaluate(survivor_genes)
				best_pop = copy.deepcopy(best_pop1)
			else:
				best_last = best_pop
		
			best_index = np.argmax(survivor_fitness)
			new_best_pop = (survivor_fitness[best_index], survivor_genes[best_index])
		
			if new_best_pop[0] > best_pop[0]:
				best_pop = new_best_pop
		
			# Insert best gene in population
			for x in range(0, len(survivor_genes), 10):
				survivor_genes[x] = best_pop[1]
				survivor_fitness[x] = best_pop[0]
			
			# Test best gene in between generations
			print("Testing best gene")
			
			for i in range(5):
				evaluate([best_pop[1]])
			print(best_pop[1])

		
			print("STD")
			print(std)
			print("Mean fit")
			print(mean)
			print("best fit")
			print(best_pop[0])

			file_aux = open( experiment_name + "/" + str(run+1) + "/results.csv", "a")
			t = env.get_time()
			file_aux.write(str(generation+1)+';'+str(best_pop[0])+';'+str(mean)+';'+str(round(std,6))+';'+str(t)+"\n")
			file_aux.close() 
			
		fim = time.time()  
		# Saving the best solutions for testing.
		np.savetxt(experiment_name + "/" + str(run+1) +'/best_solutions.txt',best_pop[1])
		# prints total execution time for experiment
		print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
		f_time = open(experiment_name + "/" + str(run+1) +"/time.txt","a")
		f_time.write('\nExecution time for the experiment: '+str(round((fim-ini)/60))+' minutes \n')
		print('\n ENDING THE EXPERIMENT \n')
	
		# Test best gene for last generation
		best_evals = []
		print("Testing best gene")
		for i in range(10):
			best_evals.append(evaluate([best_pop[1]])[0])

		file_aux = open(experiment_name + "/" + str(run+1) +"/best_fit_test.txt", "a")
		file_aux.write(str(best_evals))
		file_aux.close() 	
		
		env.save_state()
