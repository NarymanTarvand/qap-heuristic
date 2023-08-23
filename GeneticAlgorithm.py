#Import functions to be used
import numpy as np #For array manipulations
import random #for drawing random bits and setting seed
import sys #Store the biggest number



random.seed(1250778) #For consistent results



"""
CreatePopulation Function
INPUT: 
N_pop - Population size
n - number of facilities or location

OUTPUT:
X_pop - Population of random solutions

The function creates a list of solutions that are random and returns N_pop solutions with each having a cost of zero
"""
def CreatePopulation(N_pop,n):
    cost = 0 
    X_pop = []
    for i in range(N_pop):
        X_i = random.sample(range(n),n) #Sample without replacement assign n facilities to each location
        X_pop.append([X_i,cost]) #Append the solution and the default cost of zero
    return X_pop




"""
Fitness Function
INPUT:
d_kp - Distance Data matrx, already imported as numpy array
f_ij - Flow Data matrx, already imported as numpy array
X_pop - N_ x 

OUTPUT:
X_pop -  N_pop x n matrix storing all N_pop random solutions
"""
def Fitness(distance_data, flow_data, X_pop):
    population = []
    #Creates a 1xn vector from 0 to n
    set_location = np.array(range(len(X_pop[0][0]))) #Issue is X_pop[0][0] may not exist
    for i in range(len(X_pop)): #For every solution in population
        cost = 0
        X_i = np.array(X_pop[i][0]) #Store the solution instance for a solution in the population. If the form [1,5,4,2,3]

       
        for location_i, facility_i in enumerate(X_i):
            for location_j, facility_j in enumerate(X_i):
                cost = cost + flow_data[facility_i,facility_j] * distance_data[location_i,location_j] #Computes Cost
               
        X_pop[i][1] = cost #Assign updated cost for the solution instance
    
    return X_pop


"""
Mutation Function
INPUT:
offspring - solution instance of the form [[solution instance],fitness]

OUTPUT:
offspring - two elements in offspring have been swapped

The Mutation function will sweap two elements in the offspring chromosome and return the new change
"""
def Mutation(offspring):
    #offspring is of the form [[solution instance],fitness value]
    index1 = random.randint(0,len(offspring[0])-1) #Generate random number between 0 to n-1
    index2 = random.randint(0,len(offspring[0])-1)
    
    
    #Swapps two elements in the solution instance
    temp = offspring[0][index1]
    offspring[0][index1] = offspring[0][index2]
    offspring[0][index2] = temp

    return offspring




"""
Crossover Function
INPUT: 
parent1 - one solution instance of the form [[solution instance],fitness value]
parent2 - second solution instance of the form [[solution instance],fitness value]

OUTPUT
[offspring1,offspring2] - First and Second new solution produced by mating both parents
"""
def Crossover(parent1,parent2):
    Assigned_Facility_Parent1 = []
    Assigned_Facility_Parent2 = []
    offspring1 = []
    offspring2 = []
    new_offspring1 = []
    new_offspring2 = []
    cost = 0
    for i in range(len(parent1[0])):
        offspring1.append(-1) #Sets every element in offspring to -1
        offspring2.append(-1)
        
    #Store the binary bit that determines which facilities is assigned to offspring from each parent
    CrossoverMask = random.choices([0,1],k = len(parent1[0])) #Generate a n random bit either 0 or 1
     
    for j in range(len(parent1[0])):
        if CrossoverMask[j] == 1: 
            #Assign all facilities where the crossover mask is 1 into offspring1
            Assigned_Facility_Parent1.append(parent1[0][j]) #Keep track of assigned facilities so that non-filled 
                                                            #locations are not assigned by an existing placed facility
            offspring1[j] = parent1[0][j] #Assign facility at index j to offspring1 at index j
        else:
            #Assign all facilities where the crossover mask is 0 into offspring2
            Assigned_Facility_Parent2.append(parent2[0][j]) #Keep track of assigned facilities so that non-filled 
                                                            #locations are not assigned by an existing placed facility
            offspring2[j] = parent2[0][j] #Assign facility at index j to offspring1 at index j
    
    
    #In case no chromosomes are exchanged in offspring1
    Leftover_Facility1 = np.array(parent1[0])
  
    
    if len(Assigned_Facility_Parent1) != 0:
        #assigned facility is not empty
        for i in range(len(Assigned_Facility_Parent1)):
            #Remove all assigned facilities from consideration
            
            
            #First find all indexes in Leftover_Facility1 where Assigned_Facility_Parent1 occur
            #Second for every index, delete it from Leftover_Facility1         
            Leftover_Facility1 = np.delete(Leftover_Facility1, np.where(Leftover_Facility1 == Assigned_Facility_Parent1[i])[0])
            
    #In case no chromosomes are exchanged in offspring2
    Leftover_Facility2 = np.array(parent2[0])    
    if len(Assigned_Facility_Parent2) != 0:
        #assigned facility is not empty
        for i in range(len(Assigned_Facility_Parent2)):
            #Remove all assigned facilities from consideration
            
            #First find all indexes in Leftover_Facility2 where Assigned_Facility_Parent2 occur            
            #Second for every index, delete it from Leftover_Facility2
            Leftover_Facility2 = np.delete(Leftover_Facility2, np.where(Leftover_Facility2 == Assigned_Facility_Parent2[i])[0])

    
    
    
    
    #Randomise the assignment of facilities to offspring
    new_assignment_sample1 = random.sample(Leftover_Facility1.tolist(), len(Leftover_Facility1))
    new_assignment_sample2 = random.sample(Leftover_Facility2.tolist(), len(Leftover_Facility2))
    l=0 #Counters to keep track of elements in new assignment variable
    m=0
    
    for k in range(len(offspring1)):
        if(offspring1[k] == -1): #For every location where facilities are not assigned for offspring1
            offspring1[k] = new_assignment_sample1[l] #Add missing sampled facilities into offspring
            l = l + 1
        if(offspring2[k] == -1): #For every location where facilities are not assigned for offspring2
            offspring2[k] = new_assignment_sample2[m]
            m = m + 1
        
    new_offspring1.append(offspring1)
    new_offspring1.append(cost)
    
    new_offspring2.append(offspring2)
    new_offspring2.append(cost)
    
    return [new_offspring1,new_offspring2]





"""
Selection Function
INPUT:
X_pop - list of solutions from population
k - number of selected solutions to be picked before choosing the one best solution
OUTPUT:
"""
def Selection(X_pop, k):
    k_selected_solutions = random.sample(X_pop,k) #Sample without replacement k samples in the population
    k_selected_solutions.sort(key = lambda x: x[1]) #Sort the solutions in ascending order by the fitness value
    return k_selected_solutions[0]





"""
Genetic Algorithm
INPUT:
n - number of facilities or location
N_pop - initial population size
distance_data - Distance Data matrix
flow_data - Flow Data matrix
MaxIter - Maximum number of iterations to terminate the number of epochs
k - Number of solutions to select in the selection function
OUTPUT:
X_opt - Final Solution instance
The Main function to call to run the Genetic Algorithm
"""
def GeneticAlgorithm(n,N_pop,distance_data, flow_data,MaxIter,k):
    #ASSUMPTION 
    #n >> 1
    #N_pop >> 1
    initial_population = CreatePopulation(N_pop,n)
   
    current_solution = [] 
    current_objective = 0
    
    best_objective = sys.maxsize #Set best objective value to infinity
    best_solution = []
    
    
    current_generation = []
    epoch = 0 #Initial Iteration number
    
    
    #Compute Objective for all solutions in iniital population
    current_population = Fitness(distance_data,flow_data,initial_population) #ISSUE OCCURS HERE initial_poulation does not change, needs to be current_population
    current_population.sort(key = lambda x: x[1]) #Sort in ascending order
    
    while epoch < MaxIter:        
        current_solution = current_population[0][0] #Store the solution instance
        current_objective = current_population[0][1] #Store the lowest fittest value
            
        if(current_objective < best_objective):
            best_objective = current_objective #Replace best objective to the current solution
            best_solution = current_solution  #Store the best solution instance
            
        while(len(current_generation) < N_pop):
            #Generate many offspring for the next generation
            
            #Select the best solution instance out of the population to take part in mating
            parent1 = Selection(current_population,k)
            parent2 = Selection(current_population,k)
            p = random.random() #Generate random number between 0 and 1
            
            if p <= 0.8:
                #With Crossover
                #Generate two offsprings from the two parents chosen
                [offspring1,offspring2] = Crossover(parent1,parent2) 
                
                #offspring1 = [[Solution Instance],FitnessValue]
                #offspring2 = [[Solution Instance],FitnessValue]

                q = random.random() #Generate random number between 0 and 1
                if q <= 0.2:
                    #With Mutation    
                    #Mutate two elements in each offspring
                    new_offspring1 = Mutation(offspring1)
                    new_offspring2 = Mutation(offspring2)
            
                    current_generation.append([new_offspring1[0],new_offspring1[1]])
                    current_generation.append([new_offspring2[0],new_offspring2[1]])
                else:     
                    #No Mutation 
                    current_generation.append([offspring1[0],offspring1[1]])
                    current_generation.append([offspring2[0],offspring2[1]])
        
        current_population = current_generation #Store the new generation into population to evaluate fitness
        current_generation = [] #Sets new generation to be empty
        
        
        #Recompute Fitness value of new generation
        current_population = Fitness(distance_data,flow_data,current_population) 
        current_population.sort(key = lambda x: x[1]) #Sort in ascending order
        
        epoch = epoch + 1
        
    return [best_solution,best_objective]




#MAIN PROGRAM
#file_path = "Data Instance\\rou12.dat"
#[flow_data, distance_data] = read_instance_data(file_path)
#[solution,objective] = GeneticAlgorithm(np.shape(flow_data)[0],50,distance_data,flow_data,1000,10)

#print(solution)
#print(objective)













