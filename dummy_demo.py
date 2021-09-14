import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import random
import numpy as np
from math import fabs,sqrt
import glob, os
n_hidden_neurons = 10
experiment_name = 'dummy_demo'
# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
				  enemies=[2],
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
				  enemymode="static",
				  level=2,
				  speed="fastest")
		

run_mode = 'train'

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

pop_size = 100
gens = 10
mutation_chance = 0.1

# Returns list of fitness of all genes
def evaluate(genes):
	fit_list = []
	
	for gene in genes:
		env.play(pcont=gene)
		fit_list.append(fitness(env))
	return fit_list
	
from random import randrange

def fitness(env):
	return 105 - env.get_enemylife() + env.get_playerlife()


# Mates best genes with eachother to create offspring
def create_offspring(bg, bf):
	offspring = []



	shuffled = list(zip(bg, bf))
	random.shuffle(shuffled)
	bg, bf = zip(*shuffled)

	for i in range(0,len(bg),1):
		if i == len(bg) - 1:
			offspring.append((bg[i] * 0.5) + (bg[0] * 0.5))
			continue
		offspring.append((bg[i] * 0.5) + (bg[i+1] * 0.5))
	for i in range(len(offspring)):
		if mutation_chance > random.uniform(0, 1):
			offspring[i] += random.uniform(-0.5, 0.5)
			
	return offspring
	
# Selects best genes based on their fitness
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

def diversify(genes, fitness, best):
	for i in range(0, len(genes), 2):
		genes[i] = np.random.uniform(-1, 1, (1, n_vars))[0]
		
	genes[0] = best[1]
	return genes

np.set_printoptions(threshold = 10)
from numpy.random import choice


pops = np.random.uniform(-1, 1, (pop_size, n_vars))
fitness_pops = evaluate(pops)

best_genes, best_fitness = select_genes(fitness_pops, pops)
offspring_genes = create_offspring(best_genes, best_fitness)
fitness_offspring = evaluate(offspring_genes)
total_new_pop = list(offspring_genes) + list(pops)
total_fitness = fitness_offspring + fitness_pops
best_pop = np.argmax(total_fitness)
best_pop = (total_fitness[best_pop], total_new_pop[best_pop])
mean = np.mean(total_fitness)
std = np.std(total_fitness)
total_fitness = np.asarray(total_fitness)
weights = total_fitness / sum(total_fitness)
survivors_ints = choice(pop_size * 2, pop_size , p=weights, replace=False)
survivor_genes = [total_new_pop[i] for i in survivors_ints]
survivor_fitness = [total_fitness[i] for i in survivors_ints]

best_pop = np.argmax(total_fitness)
best_pop = (total_fitness[best_pop], total_new_pop[best_pop])
best_last = best_pop

print("STD")
print(std)
print("Mean fit")
print(mean)
print("best fit")
print(best_pop[0])

for generation in range(gens):
	print("New generation")
	pops = survivor_genes
	fitness_pops = survivor_fitness
	best_genes, best_fitness = select_genes(fitness_pops, pops)
	offspring_genes = create_offspring(best_genes, best_fitness)
	fitness_offspring = evaluate(offspring_genes)
	
	total_new_pop = list(offspring_genes) + list(pops)
	total_fitness = fitness_offspring + fitness_pops
	
	best_pop = np.argmax(total_fitness)
	best_pop = (total_fitness[best_pop], total_new_pop[best_pop])
	
	mean = np.mean(total_fitness)
	std = np.std(total_fitness)
	
	total_fitness = np.asarray(total_fitness)
	weights = total_fitness / sum(total_fitness)
	survivors_ints = choice(pop_size * 2, pop_size , p=weights, replace=False)
	
	survivor_genes = [total_new_pop[i] for i in survivors_ints]
	survivor_fitness = [total_fitness[i] for i in survivors_ints]
	
	
	if best_pop[0] > best_last[0]:
		best_last = best_pop
	else:
		print("Adding new genes@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		survivor_genes = diversify(survivor_genes, survivor_fitness, best_pop)
		survivor_fitness = evaluate(survivor_genes)
		
	print("STD")
	print(std)
	print("Mean fit")
	print(mean)
	print("best fit")
	print(best_pop)









