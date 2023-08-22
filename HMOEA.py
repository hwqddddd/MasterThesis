import os
import io
import random
import numpy as np
import fnmatch
import csv
import array
import scipy.stats
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from csv import DictWriter
from json import load, dump
from deap import base, creator, tools, algorithms, benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume, igd
from deap.tools._hypervolume import hv
from glob import glob


def load_instance(json_file):
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None


def routeToSubroute(individual, instance):
    route = []
    sub_route = []
    vehicle_load = 0
    vehicle_capacity = instance['vehicle_capacity']
    
    for customer_id in individual:

        demand = instance[f"customer_{customer_id}"]["demand"]

        new_vehicle_load = vehicle_load + demand

        if(new_vehicle_load <= vehicle_capacity):
            sub_route.append(customer_id)
            vehicle_load = new_vehicle_load
        else:
            route.append(sub_route)
            sub_route = [customer_id]
            vehicle_load = demand


    if sub_route != []:
        route.append(sub_route)

    # Returning the final route with each list inside for a vehicle
    return route

def getNumVehiclesRequired(individual, instance):
    subroute = routeToSubroute(individual, instance)
    num_of_vehicles = len(subroute)

    return num_of_vehicles

def getRouteCost(individual, instance):
    total_cost = 0
    updated_route = routeToSubroute(individual, instance)

    for sub_route in updated_route:
        # Initializing the subroute distance to 0
        # sub_route_distance = 1
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0
        # last_customer_id = 1

        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = instance["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id
        
        sub_route_distance = sub_route_distance + instance["distance_matrix"][last_customer_id][0]

        total_cost = total_cost + sub_route_distance
    
    return total_cost

def getRouteTime(individual, instance):
    total_time = 0
    waiting_time = 0
    delay_time = 0
    sub_route_max = 0
    updated_route = routeToSubroute(individual, instance)
    for sub_route in updated_route:

        sub_route_time = 0

        last_customer_id = 0

        for customer_id in sub_route:

            time = instance["distance_matrix"][last_customer_id][customer_id]
            
            arrival_time = total_time + time
            
            ready_time = instance[f"customer_{customer_id}"]["ready_time"]
            service_time = instance[f"customer_{customer_id}"]["service_time"]
            due_time = instance[f"customer_{customer_id}"]["due_time"]

            if arrival_time < ready_time:
                waiting_time += ready_time - arrival_time
                total_time += waiting_time
            elif arrival_time > due_time:
                delay_time += arrival_time - due_time
                total_time += delay_time
            total_time += service_time + time
            sub_route_time += time
            # Update last_customer_id to the new one
            last_customer_id = customer_id

        sub_route_time = sub_route_time + instance["distance_matrix"][last_customer_id][0]
        if sub_route_time > sub_route_max:
            sub_route_max = sub_route_time


        return total_time, sub_route_max, waiting_time, delay_time

def eval_indvidual_fitness(individual, instance):

    vehicles = getNumVehiclesRequired(individual, instance)

    total_time, sub_route_max, waiting_time, delay_time = getRouteTime(individual, instance)

    return (vehicles, total_time, sub_route_max, waiting_time, delay_time)

def similarity(ind1, ind2):
    n = len(ind1)
    arc_matrix1 = [[0 for _ in range(n)] for _ in range(n)]
    arc_matrix2 = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n-1):
        arc_matrix1[ind1[i]-1][ind1[i+1]-1] = 1
    for i in range(n-1):
        arc_matrix2[ind2[i]-1][ind2[i+1]-1] = 1
    common = 0
    total = 2*n - 2
    for i in range(n):
        for j in range(n):
            if arc_matrix1[i][j] == 1 and arc_matrix2[i][j] == 1:
                common += 1
                total -= 1
    return common / total

def cxOrdered(input_ind1, input_ind2):
    ind1 = [x-1 for x in input_ind1]
    ind2 = [x-1 for x in input_ind2]

    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    flag1, flag2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            flag1[ind2[i]] = False
            flag2[ind1[i]] = False

    temp1, temp2 = ind1, ind2
    m1, m2 = b + 1, b + 1
    for i in range(size):
        if not flag1[temp1[(i + b + 1) % size]]:
            ind1[m1 % size] = temp1[(i + b + 1) % size]
            m1 += 1

        if not flag2[temp2[(i + b + 1) % size]]:
            ind2[m2 % size] = temp2[(i + b + 1) % size]
            m2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Finally adding 1 again to reclaim original input
    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutationShuffle(individual, mrate):
    size = len(individual)
    for i in range(size):
        if random.random() < mrate:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = individual[swap_indx], individual[i]

    return individual

def TwoOpt(pop,instance):
    ind = random.sample(pop,1)[0]

    sub = routeToSubroute(ind,instance)
    best = ind

    x = 0
   
    while x<5:
        x += 1
        sub = routeToSubroute(ind,instance)
        i = random.randint(0,len(sub)-1)
        j = random.randint(0,len(sub)-1)
        new = sub[:] 
        a = random.randint(0,len(sub[i])-1)
        b = random.randint(0,len(sub[j])-1)
        new[i][a:] = sub[j][b:]
        new[j][b:] = sub[i][a:]
        temp = []
        for i in new:
            temp = temp+i
        if getRouteCost(temp, instance) < getRouteCost(ind, instance):
            best = temp
            improved = True
            ind = best

    return best

class HMOEA(object):

    def __init__(self,datapath):
        self.json_instance = load_instance(datapath)
        # Amazon data ind_size - 1
        self.ind_size = self.json_instance['Number_of_customers'] - 1
        # self.ind_size = self.json_instance['Number_of_customers'] 
        self.pop_size = 100
        self.cross_prob = 0.85
        self.mut_prob = 0.1
        self.num_gen = 100
        self.toolbox = base.Toolbox()
        self.createCreators()

    def createCreators(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # Registering toolbox
        self.toolbox.register('indexes', random.sample, range(1, self.ind_size + 1), self.ind_size)

        # Creating individual and population from that each individual
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register('evaluate', eval_indvidual_fitness, instance=self.json_instance, unit_cost=1)

        # Selection method
        # NSGA2
        self.toolbox.register("select", tools.selNSGA2)

        # Crossover method
        self.toolbox.register("mate", cxOrdered)

        # Mutation method
        self.toolbox.register("mutate", mutationShuffle, mrate=self.mut_prob)
        # self.toolbox.register("mutate", mutation, indpb=self.mut_prob)


    def generatingPopFitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_ind))

        for ind, fit in zip(self.invalid_ind, self.fitnesses):
            ind.fitness.values = fit
        # ref_points = tools.uniform_reference_points(nobj=5, p=12)
        self.pop = self.toolbox.select(self.pop, len(self.pop))
   
    def runGenerations(self, S_M = True, L_S = True):
        # Running algorithm for given number of generations
        start_time = time.time()
        for _ in range(self.num_gen):

            self.offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            self.offspring = [self.toolbox.clone(ind) for ind in self.offspring]

            # Using similarity measure to selcet the lowest similarity individual from 5 ind to crossover

            if S_M:
                for ind1 in self.offspring:
                    simi = 1
                    for i in random.sample(self.offspring,5):
                        temp = similarity(ind1, i)
                        if temp < simi:
                            simi = temp
                            ind2 = i
                    if random.random() <= self.cross_prob:
                        self.toolbox.mate(ind1, ind2)


                        del ind1.fitness.values, ind2.fitness.values
                    self.toolbox.mutate(ind1)
                    self.toolbox.mutate(ind2)
            else:
                for ind1, ind2 in zip(self.offspring[::2], self.offspring[1::2]):
                    if random.random() <= self.cross_prob:
                        self.toolbox.mate(ind1, ind2)

                        del ind1.fitness.values, ind2.fitness.values
                    self.toolbox.mutate(ind1)
                    self.toolbox.mutate(ind2)

            # Calculating fitness for all the invalid individuals in offspring
            self.invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.invalid_ind)
            for ind, fit in zip(self.invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Use 2 opt to optimize the individual
            if L_S:
                TwoOpt(self.offspring, self.json_instance)

            self.pop = self.toolbox.select(self.pop + self.offspring, self.pop_size)

        print("--- %.2f seconds ---" % (time.time() - start_time))
        
        # Normalize the fitness
        fitness_values = [0] * 5
        for i in range(5):
            fitness_values[i] = [p.fitness.values[i] for p in self.pop]
           
        for ind in self.pop:
            for i, fitness_value in enumerate(ind.fitness.values):
                # Calculate the range of fitness values for this objective in the population
                # fitness_values = [p.fitness.values[i] for p in self.pop]
                fitness_range = max(fitness_values[i]) - min(fitness_values[i])
                if fitness_range == 0:
                    fitness_range += 1
                # Normalize the fitness value of this individual for this objective to the range [0, 1]
                ind.fitness.values = ind.fitness.values[:i] + ((fitness_value - min(fitness_values[i])) / fitness_range,) + ind.fitness.values[i+1:]
        front = tools.sortNondominated(self.pop, 100, first_front_only=True)[0]

        wobj = np.array([ind.fitness.wvalues for ind in front]) * -1


        hpv = hv.hypervolume(wobj, [1.01, 1.01, 1.01, 1.01, 1.01])

    
        print(f"{20 * '#'} End of Generations {20 * '#'} ")
        print("Hypervolume: ",hpv)
        # print('IGD', IGD)
        return hpv, front
       



    def runMain(self, S_M = True, L_S = True):
        self.generatingPopFitness()
        hpv, front = self.runGenerations(S_M, L_S)
        self.getBestInd()
        self.doExport()
        return hpv, front
    

# deviation distribution
def dominates(obj1, obj2):
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
def calculate_ratio(former_population, latter_population):
    dominated_count = 0
    for latter_solution in latter_population:
        for former_solution in former_population:
            if dominates(former_solution.fitness.values, latter_solution.fitness.values):
                dominated_count += 1
                break  # Move to the next latter_solution

    return dominated_count / len(latter_population)


if __name__ == "__main__":
    
    N_hpv_sum = []
    NH_C_sum = []
    H_hpv_sum = []
    HN_C_sum = []
    for _ in range(10):
        for i in glob('./data/Amazon_data/TW1_json/A200*'):
        # for i in glob('./data/json/RC208*'):
            print(i)
            print(f"{20 * '#'} ",i[12:16], f"{20 * '#'} ") 

            NSGA = HMOEA(i)
            N_hpv, N_front = NSGA.runMain(S_M = False, L_S = False)
            # print(someinstance.runMain())
            N_hpv_sum.append(N_hpv)
            del NSGA
            del creator.FitnessMin
            del creator.Individual

            HMOEA = HMOEA(i)
            H_hpv, H_front = HMOEA.runMain()
            H_hpv_sum.append(H_hpv)
            del HMOEA
            del creator.FitnessMin
            del creator.Individual


            c_metric_HN = calculate_ratio(H_front, N_front)
            c_metric_NH = calculate_ratio(N_front, H_front)
            NH_C_sum.append(c_metric_NH)
            HN_C_sum.append(c_metric_HN)
    print(N_hpv_sum)
    print('N_hpvavg', np.mean(N_hpv_sum))
    print('N_hpvstd', np.std(N_hpv_sum,ddof=1))
    print(H_hpv_sum)
    print('H_hpvavg', np.mean(H_hpv_sum))
    print('H_hpvstd', np.std(H_hpv_sum,ddof=1))
    print(NH_C_sum)
    print('NH_C_sumavg', np.mean(NH_C_sum))
    print('NH_C_sumstd', np.std(NH_C_sum,ddof=1))
    print(HN_C_sum)
    print('HN_C_sumavg', np.mean(HN_C_sum))
    print('HN_C_sumstd', np.std(HN_C_sum,ddof=1))





