import csv
import os, re
import pandas
import numpy
from math import *
import datetime
import time
import gmplot
import geocoder
import random
import matplotlib.pyplot as plt
import networkx as nx
import copy, functools
from functools import partial
from datetime import datetime
startTime = datetime.now()


### IMPORT DATA ###

#distanceMatrix = # import distance matrix
#timeMatrix = # import time matrix
#initialSolution_original = # import initial solution

### SET VARIABLES ###

# ggeneral variables
store_id = 0
store_id2 = 1133 # 260
max_route_duration = 600 #!!!!!!!!
#service_time= 0

# SA variables
temperature=100
beta=2   # bias
max_iterations= 1000 #400 !!
alpha=0.95
NSize = 4 # Neighborhood Size
max_per_temperature_level = 2

# LNS variables
p = 4 # number of customers
maxiter1 = 3
maxiter2 = 20
max_no_improvement = 10
sizeLNS = 26

### BUILD ARCS\EDGES MATRIX ###

def buildArcs(initialSolution_original):

    # Build arcs\edges

    initialSolution_arcs = initialSolution_original[1:]

    sequence = list(initialSolution_arcs['OrderID'])

    # build i nodes
    arcs_i = []
    for i in range(len(initialSolution_arcs)):
        if initialSolution_arcs.iloc[i,1] == initialSolution_arcs.iloc[i-1,1]:
            arcs_i.append(sequence[i])
        else:
            #arcs_i.append(0)
            arcs_i.append(store_id2)
            arcs_i.append(0)
            arcs_i.append(sequence[i])

    # build j nodes
    arcs_j = arcs_i[1:] #select from 2nd element on
    arcs_j.insert(len(arcs_i),arcs_i[0]) #insert the first element of list i at last

    # create dataframe with i and j
    arcs = pandas.DataFrame( {'i': pandas.Series(arcs_i),
                             'j': pandas.Series(arcs_j)})

    return arcs


def adjacencyMatrix(initialSolution_original):

    # Build adjacency matrix
    nodes = initialSolution_original['OrderID'].tolist()
    nodes.append(store_id2) # add final depot
    G = nx.Graph() # create empty graph
    G.add_nodes_from(nodes) # add nodes to empty graph

    # create arcs df
    arcs = buildArcs(initialSolution_original)

    # create a list of edges
    edgesList =[]
    # Iterate over each row
    for index, rows in arcs.iterrows():
        # Create list for the current row
        my_list = (int(rows.i), int(rows.j))
        # append the list to the final list
        edgesList.append(my_list)

    G.add_edges_from(edgesList) # add the edges to the graph
    #print(G.edges())

    arcs_mtx_original = nx.to_numpy_matrix(G) # create adjency matrix as numpy array
    arcs_mtx_original = pandas.DataFrame(arcs_mtx_original) # as data frame

    # change index to the real custumer/node number
    nodes = pandas.Series(nodes) # First change the list to series. My index is the same than list nodes
    arcs_mtx_original = arcs_mtx_original.set_index(nodes) # change index
    arcs_mtx_original.columns = nodes # change for column names
    # Delete conexion store_id -> sore_id2 (delete conexion in depot that was created when building the arcs)
    arcs_mtx_original.loc[store_id, store_id2] = 0
    arcs_mtx_original.loc[store_id2, store_id] = 0

    return arcs_mtx_original

### INPUT FOR FUNCTIONS THAT FIND SUCCESOR AND PREDECESOR NODES ###

# make a list with last customer/node of each route/trip
trip =  initialSolution_original["vehicleNumber"].tolist()
id = initialSolution_original['OrderID'].tolist()

lastNodeOfRoutes = []

for i in initialSolution_original["id"]:
    if  trip[i] != trip[i-1]:
        lastNodeOfRoutes.append(id[i-1])


def successor(arcs_mtx,i):
    succ = arcs_mtx.columns[(arcs_mtx.loc[i,:]==1)]

    if len(succ)==0:
        print('')
    else: return succ[1]


def predecessor(arcs_mtx,i):
    pred = arcs_mtx.columns[(arcs_mtx.loc[i,:]==1)]

    if len(pred) == 0:
        print('')
    else: return pred[0]


def solution(arcs_mtx, printroute=False):

    arcs_mtx1 = arcs_mtx.copy()

    # 1) Initialize solution matrix
    sol = pandas.DataFrame(index= customersplus,columns= ['OrderID','Start_Time','End_Time','route','stop_number','arrival_time','drive_time','drive_distance','delivery_start','infeasible_flag','route_start_time','route_end_time','return_distance', 'route_duration', 'capacity', 'load', 'accumul_load'])

    #fill  order id and TW
    for ii in customers:
        for col in ['OrderID','Start_Time','End_Time', 'load', 'capacity']:
        #for col in ['OrderID','Start_Time','End_Time', 'route']:
            sol.loc[ii,col]= initialSolution[initialSolution["OrderID"] == ii][col].iloc[0]

    sol['infeasible_flag'] = sol['route'] = 0 # I don'' understand why this!!
    sol.loc[store_id,'Start_Time'] = sol.loc[store_id,'arrival_time']= 0 # TW (start time and arrival time) for the depot
    sol.loc[store_id,'End_Time'] = 1109 # TW (end time) for the depot

    # 2) create the solution and check feasibility

    for route_number in range(1,row_count+1):

        # find route head
        for i in customers:
            if (arcs_mtx1.loc[store_id,i]==1) and (sol.loc[i,'route']==0) and (successor(arcs_mtx1, i)!=0):  #skip the ones that are already visited or are not linked to depo
                break
            elif i == customers[row_count-1]:
                printroute = True
                if printroute: print('\nComplete!')

        if (i == customers[row_count-1]) and (sol.loc[i,'route']!=0):
            break

        if printroute: print('\nroute', route_number, ': ')

        prev = store_id
        stop_number = 1

        #loop throught the route & find successors
        for stop_number in range(1,row_count+1):
            printroute = True
            if printroute: print(i),
            sol.loc[i,'route']= route_number

            #calculate variables
            sol.loc[i,'drive_time']= drive_time.iloc[prev,i]
            sol.loc[i,'drive_distance']= dist_mtx.iloc[prev,i]

            if stop_number==1:
                sol.loc[i,'route_start_time']= sol.loc[i,'Start_Time']-drive_time.iloc[prev,i]-service_time
                sol.loc[i,'arrival_time']= sol.loc[i,'route_start_time']+ drive_time.iloc[prev,i]
                route_start_time = sol.loc[i,'route_start_time'] #save start time
                if route_start_time < 0:
                        sol.loc[i,'route_start_time']= 0
                        sol.loc[i,'arrival_time']= drive_time.iloc[prev,i] + service_time
                        route_start_time = sol.loc[i,'route_start_time'] #save start time
            else:
                sol.loc[i,'arrival_time']= sol.loc[prev,'arrival_time']+ drive_time.iloc[prev,i] + service_time

            #time constraints
            delivery_start = max(sol.loc[i,'arrival_time'],sol.loc[i,'Start_Time'])
            sol.loc[i,'delivery_start'] = delivery_start
            if delivery_start > sol.loc[i,'End_Time']:  sol.loc[i,'infeasible_flag']=1 #check feasibility

            #capacity constraints
            accumul_load = sol.loc[sol['route'] == route_number, 'load'].sum()
            sol.loc[i,'accumul_load'] = accumul_load
            if accumul_load > sol.loc[i,'capacity']: sol.loc[i,'infeasible_flag']=1

            #stop number
            sol.loc[i,'stop_number'] = stop_number

            prev = i
            i = successor(arcs_mtx1,i)

            #find route end
            if i == store_id2:
                sol.loc[prev,'route_end_time'] = sol.loc[prev,'arrival_time'] + drive_time.iloc[prev,i] + service_time
                sol.loc[prev,'return_distance'] = dist_mtx.iloc[prev,i] # distance to return to depot
                sol.loc[prev,'route_duration'] = sol.loc[prev,'route_end_time']-route_start_time
                if sol.loc[prev,'route_duration'] > max_route_duration:  sol.loc[prev,'infeasible_flag'] = 1 #check feasibility
                break

    return sol


def latestArrivalTime(solution_mtx, arcs, i):
    # rename
    solution_mtx_zplus = solution_mtx
    arcs_z = arcs

    # find i successor
    i_successor = successor(arcs_z,i)

    # index to find the row where the customers i and i_successor are located on the table "solution"
    indexRow_i = solution_mtx_zplus.index[solution_mtx_zplus['OrderID'] == i]
    indexRow_i = indexRow_i[0]

    if i_successor == store_id2: # if i_successor is the last customer in the route -> after that drives to depot
        z = min(solution_mtx_zplus.loc[indexRow_i,'route_end_time'] - drive_time.iloc[i,i_successor] - service_time, solution_mtx_zplus.loc[indexRow_i, 'End_Time'])
    else:
        # find indexes in order to access to the tables
        indexRow_iSuccesor = solution_mtx_zplus.index[solution_mtx_zplus['OrderID'] == i_successor]
        indexRow_iSuccesor = indexRow_iSuccesor[0]
        z = min(solution_mtx_zplus.loc[indexRow_iSuccesor,'arrival_time'] - drive_time.iloc[i,i_successor] - service_time, solution_mtx_zplus.loc[indexRow_i, 'End_Time'])

    return z


# Objective Function LNS
def objectiveFunction(solution_mtx):

    # rename
    solution_mtx_of = solution_mtx

    # objective function
    total_distance = solution_mtx_of['drive_distance'].sum(skipna=True) + solution_mtx_of['return_distance'].sum(skipna=True)
    total_cost = total_distance

    return total_cost

# Objective function SA: 1st element
def costNumberOfRoutes(solution_mtx):

    # rename
    solution_mtx_1st = solution_mtx

    NRoutes = solution_mtx_1st.route.max()

    return NRoutes

# Objective function SA: 2nd element
def costStopsDistribution(solution_mtx):

    # rename
    solution_mtx_2nd = solution_mtx
    solution_mtx_2nd = solution_mtx_2nd[1:] # remove first row because is the depot

    NRoutes = solution_mtx_2nd.route.max() # identify number of routes

    stops = list() # empty list for adding number of stops

    # find number of routes
    for r in range(1,NRoutes+1):
        stops_on_route = solution_mtx_2nd[solution_mtx_2nd.route == r].stop_number.max()
        stops.append(stops_on_route)

    # calculate the distribution of stops
    stops_distribution = sum([i**2 for i in stops])
    stops_distribution = stops_distribution * -1

    return stops_distribution

# Objective function SA: 3rd element of objective function
def costMDL(solution_mtx, arcs):

    # rename
    solution_mtx_cost = solution_mtx
    solution_mtx_cost = solution_mtx_cost.iloc[1:] # remove first row because is the depot
    arcs_cost = arcs

    # some preprocessing
    solution_mtx_cost = solution_mtx_cost.sort_values(['route', 'stop_number'], ascending=[True, True])
    # reset index
    solution_mtx_cost = solution_mtx_cost.reset_index() # to find jplus by looking into the nex row instead of calling function "succesor"

    # identify number of routes
    NRoutes = solution_mtx_cost.route.max()

    stops = list() # list with number of stops

    for r in range(1,NRoutes+1):
        stops_on_route = solution_mtx_cost[solution_mtx_cost.route == r].stop_number.max()
        stops.append(stops_on_route)

    shortestRoute = stops.index(min(stops)) + 1 # plus 1 because the first route number starts in 1

    # compute PLAN(sigma) minimal delay (mdl)

    # create data frame with the total load of each route
    routes_load = pandas.DataFrame(solution_mtx_cost.groupby('route')['load'].sum())
    # create data frame with the capacity of the vehicle
    routes_capacity = pandas.DataFrame(solution_mtx_cost.groupby('route')['capacity'].max())
    # create data frame with the time at which the route ended (at the depot)
    route_start = pandas.DataFrame(solution_mtx_cost.groupby('route')['route_start_time'].min())

    # create empty lists to store results
    mdl_list = [] # for intermediate results
    mdl_customers = [] # for final mdl for each customer

    # just a high number to compare and store the lowest mdl found so far
    lowest_mdl = 10**100

    # set te value of infinite
    infinite = 10**6

    # compute mdl
    for i in solution_mtx_cost["OrderID"]:

        indexRow_i = solution_mtx_cost.index[solution_mtx_cost['OrderID'] == i] #
        indexRow_i = indexRow_i[0]


        if solution_mtx_cost.loc[indexRow_i,"route"] == shortestRoute: # if i in the smallest route

            for j in solution_mtx_cost["OrderID"]:

                # find index for j
                indexRow_j = solution_mtx_cost.index[solution_mtx_cost['OrderID'] == j] #
                indexRow_j = indexRow_j[0]

                if solution_mtx_cost.loc[indexRow_i,"route"] != solution_mtx_cost.loc[indexRow_j,"route"]: # if j is in another route other than i's route

                    # identify j's route number
                    routeNumber_j = solution_mtx_cost.loc[indexRow_j,"route"]

                    # find out whether inserting i in the route will exceed the capacity of the vehicle
                    if (solution_mtx_cost.loc[indexRow_i,"load"] + routes_load.loc[routeNumber_j,'load']) > routes_capacity.loc[routeNumber_j,'capacity']:
                        mdl_list.append(infinite)

                    else:
                        # find j_plus and its index (to find its TW)
                        jPlus = successor(arcs_cost, j)

                        # compute deltas
                        delta_j = solution_mtx_cost.loc[indexRow_j, 'arrival_time'] + service_time # departure time j
                        delta_i = delta_j + drive_time.iloc[i,j] + service_time # departure time i

                        # compute minimal delay of customer i when j is the last node in the route (and therefore jPlus the depot)
                        if jPlus == store_id2:
                            mdl = max(delta_j + drive_time.iloc[i,j] - solution_mtx_cost.loc[indexRow_i,'End_Time'], 0) + max(delta_i + drive_time.iloc[i,jPlus] - (route_start.loc[routeNumber_j, 'route_start_time'] + solution_mtx_cost.loc[indexRow_j,'route_end_time']), 0)
                            # mdl_list.append(mdl)
                        else: # compute minimal delay of customer i (when j is any other place other than the last of the route)
                            indexRow_jPlus = solution_mtx_cost.index[solution_mtx_cost['OrderID'] == jPlus]
                            indexRow_jPlus = indexRow_jPlus[0]

                            mdl = max(delta_j + drive_time.iloc[i,j] - solution_mtx_cost.loc[indexRow_i,'End_Time'], 0) + max(max(delta_i + drive_time.iloc[i,jPlus], solution_mtx_cost.loc[indexRow_jPlus,'Start_Time']) - latestArrivalTime(solution_mtx_cost,arcs_cost,jPlus), 0)

                        if mdl < lowest_mdl: # choose the lowest minimal delay (to store only the minimal found so far)
                            lowest_mdl = mdl
                        elif mdl == 0: # if mdl = 0 stop searching
                            break

            mdl_list.append(lowest_mdl) # append the lowest mdl found so far
            # final_mdl_customer = min(mdl_list)
            mdl_customers.append(min(mdl_list)) # append the minimum mdl of each customer
            # empty the list for the next i iteration
            mdl_list = []
            # set lowest_mdl to start the comparison again
            lowest_mdl = 10**100

    mdl_plan = sum(mdl_customers)

    return mdl_plan


def feasible(solution_mtx):
    f = solution_mtx['infeasible_flag'].sum(skipna=True)
    if f==0: return 1
    elif f>0: return 0
    else: return 'Error'


def relocate(arcs_mtx,i,j):

    arcs_new = arcs_mtx.copy()

    # Take i from where it was located
    pred_i = predecessor(arcs_new,i)
    succ_i = successor(arcs_new,i)
    succ_j = successor(arcs_new, j)

    if pred_i != succ_i:
        arcs_new.loc[pred_i,succ_i]=1  # create an arc pred(i) -> succ(i) ; only if both are not the depo
        arcs_new.loc[succ_i,pred_i]=1

    arcs_new.loc[pred_i, i]=0 # remove the arcs: pred(i) -> i
    arcs_new.loc[i, pred_i]=0 # remove the arcs: pred(i) -> i

    arcs_new.loc[i,succ_i]=0 # remove: i -> succ(i)
    arcs_new.loc[succ_i, i]=0 # remove: i -> succ(i)


    # insert i after j
    succ_j = successor(arcs_new,j)
    arcs_new.loc[j,succ_j]=0 # remove edge j -> succ(j)
    arcs_new.loc[succ_j,j]=0 # remove edge j -> succ(j)
    arcs_new.loc[j,i]=1 # add edge j -> i
    arcs_new.loc[i,j]=1 # add edge j -> i
    arcs_new.loc[i,succ_j]=1 # add edge i -> succ(j)
    arcs_new.loc[succ_j,i]=1 # add edge i -> succ(j)

    columnsList = list(arcs_new.columns)
    columnsList.index(j)
    position_i = columnsList.index(i)
    position_j = columnsList.index(j)

    if position_j < position_i: # relocate column i to be after j
        change = list(arcs_new[i]) # store column i in a list apart
        arcs_new = arcs_new.drop(i, 1) # delete column i from df
        arcs_new.insert(position_j+1, column = i, value = change) # relocate column i

    if position_j > position_i:
        change = list(arcs_new[i]) # store column i in a list apart
        arcs_new = arcs_new.drop(i, 1) # delete column i from df
        arcs_new.insert(position_j, column = i, value = change) # relocate column i (this changes compared to the above case)

    return arcs_new.astype(int)

def Initialize():

    global zips, allorders, orders, customers, customersplus, row_count, arcs, store_id, row_count, drive_time, dist_mtx, tg1removal, initialSolution, arcs_mtx

    # call data
    initialSolution = initialSolution_original

    #customers
    customers = initialSolution['OrderID'].tolist()
    customers = customers[1:] # ommit first row because it is the depot
    customers = list(set(customers)) #only sorts in ascendent order
    row_count = len(customers)

    #customersplus
    customersplus = [store_id]+ customers # add depot

    # distance matrix
    dist_mtx = distanceMatrix

    # time matrix
    drive_time = timeMatrix

    # maximum distance
    max_distance = dist_mtx.values.max()

    # arcs matrix
    arcs_mtx = adjacencyMatrix(initialSolution)


def neighborhood(arcs_mtx,size):
    global customers

    neighbors=list()

    ns = size

    arcs_neighbor = arcs_mtx.copy() #initialize

    sample = random.sample(customers, 1+ns) #random select

    for j in sample[1:ns+1]:
        i = sample[0]
        relocated=relocate(arcs_neighbor,i,j).copy()
        arcs_neighbor=relocated
        neighbors.append(arcs_neighbor)

        print('.')

    return neighbors

"""
SIMULATED ANNEALING ALGORITHM (SA)
"""

    Initialize()

    originalNSize = NSize

    best_arcs = arcs_mtx

    best_solution = solution(best_arcs)

    best_cost_1stElement = costNumberOfRoutes(best_solution) # cost related to number of routes
    best_cost_2ndElement = costStopsDistribution(best_solution) # cost related to stops distribution
    best_cost_3rdElement = costMDL(best_solution,best_arcs) # cost related to mdl

    current_arcs = best_arcs.copy()

    current_solution_mtx = best_solution

    costs_achieved = [best_cost_1stElement, best_cost_2ndElement, best_cost_3rdElement]

    no_improvement = 0
    empty_counter = 0
    no_improvement_short_route = 0
    shortest_route_index = 0
    iteration_counter = 0
    iteration_counter2 = 0

    everything_1st = [] # to store all feasible solutions (1st element)
    everything_2nd = [] # to store all feasible solutions (2nd element)
    everything_3rd = [] # to store all feasible solutions (3rd element)


    for iteration in range(max_iterations):

        iteration_counter += 1

        if temperature <= 0.01:
            print('finished due to the minimal temperature has been reached')
            break

        for i in range(max_per_temperature_level):

            if no_improvement > max_no_improvement_sa:
                print('Number of iterations with no improvement', no_improvement)

            try:
                print('Iteration',iteration)
                if no_improvement > max_no_improvement_sa and iteration < max_iterations/2: NSize = min(NSize+4,12) #expand search

                elif empty_counter/iteration_counter >= .6:
                    NSize = min(NSize+4,12) # expand search
                    empty_counter = 0 # reset counter
                    iteration_counter = 0 # reset counter

                else: NSize = originalNSize #narrow search

                # check number of visits in each route
                solution_matrix = current_solution_mtx.iloc[1:] # exclude the node that is the depot
                number_of_routes = solution_matrix.groupby(by = 'route')['stop_number'].max()
                number_of_routes = number_of_routes.to_frame()
                number_of_routes = number_of_routes.sort_values(['stop_number'])

                # compute what can be considered as a very short route
                short_route_limit = int(number_of_routes.mean() - number_of_routes.std())

                if iteration_counter2 == 10:
                    no_improvement_short_route = 0 # reset and try again with short routes
                    iteration_counter2 = 0 #reset

                # Favor the relocation of nodes in the shortest routes
                if no_improvement_short_route <= 10 and number_of_routes.iloc[shortest_route_index,0] <= short_route_limit:
                    customers_shortest_route = solution_matrix[solution_matrix['route'] == number_of_routes.index[shortest_route_index]]
                    customers_shortest_list = list(customers_shortest_route['OrderID'])

                    l = neighborhoodB(current_arcs, customers_shortest_list)   #create neighborhood

                    l_solutions = []
                    l_feasible = []

                    for i in l:
                        l_solve = solution(i)
                        l_solutions.append(l_solve)
                        l_feas = feasible(l_solve)
                        l_feasible.append(l_feas)

                    # store only feasible solutions
                    l_solutions_new = []
                    l_new = []

                    for i in range(len(l)):
                        if l_feasible[i] == 1:
                            l_solutions_new.append(l_solutions[i])
                            l_new.append(l[i])

                    l_solutions = l_solutions_new # rename
                    l = l_new # rename

                    # If no feasible neighbor
                    if len(l) == 0:
                        #print '(neighborhood empty)'
                        print('(neighborhood empty)')
                        empty_counter+=1
                        #no_improvement_short_route +=1
                        continue #

                    # compute cost of feasible (1st element) neighbors and rank them
                    rank_first_elem = map(costNumberOfRoutes,l_solutions) #compute costs
                    rank_first_elem = pandas.DataFrame(rank_first_elem, columns = ['cost_NumberOfRoutes']) #convert to DF
                    rank_first_elem.sort_values('cost_NumberOfRoutes', ascending = True, inplace=True) #rank neighbors
                    numberOfRoutes_cost = rank_first_elem.iloc[0]['cost_NumberOfRoutes']

                    everything_1st.append(numberOfRoutes_cost)

                    #if delta >  0:
                    if numberOfRoutes_cost <  best_cost_1stElement:
                        print('Cost improved (1st element) --->', numberOfRoutes_cost)
                        best_cost_1stElement = numberOfRoutes_cost
                        current_cost = [['improvement 1st element'],[best_cost_1stElement],[datetime.now() - startTime]] # to store, track and plot
                        best_arcs = l[rank_first_elem.index[0]].copy()
                        current_arcs = l[rank_first_elem.index[0]].copy()
                        current_solution_mtx = l_solutions[rank_first_elem.index[0]]

                        # also store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                        best_cost_2ndElement = costStopsDistribution(current_solution_mtx)
                        best_cost_3rdElement = costMDL(current_solution_mtx,best_arcs)

                        # reset no_improvement counter
                        no_improvement = 0
                        no_improvement_short_route = 0

                    else: # else compute second element cost
                        rank_second_elem = map(costStopsDistribution,l_solutions) #compute costs
                        rank_second_elem = pandas.DataFrame(rank_second_elem, columns = ['cost_stopsDistribution']) #convert to DF
                        rank_second_elem.sort_values('cost_stopsDistribution', ascending = False, inplace = True)
                        stopsDistribution_cost = (rank_second_elem.iloc[0]['cost_stopsDistribution']) # take the highest value

                        #store feasible solutions
                        everything_2nd.append(stopsDistribution_cost)

                        #if delta > 0:
                        if stopsDistribution_cost < best_cost_2ndElement:
                            print('Cost improved (2nd element) --->', stopsDistribution_cost)
                            best_cost_2ndElement = stopsDistribution_cost
                            current_cost = [['improvement 2nd element'],[best_cost_2ndElement],[datetime.now() - startTime]] # to store, track and plot
                            best_arcs = l[rank_second_elem.index[0]].copy()
                            current_arcs = l[rank_second_elem.index[0]].copy()
                            current_solution_mtx = l_solutions[rank_second_elem.index[0]]

                            # also compute and store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                            best_cost_1stElement = costNumberOfRoutes(current_solution_mtx) # already calculated before
                            best_cost_3rdElement = costMDL(current_solution_mtx,best_arcs)

                            # reset no_improvement counter
                            no_improvement = 0
                            no_improvement_short_route = 0

                        else: # compute 3rd element cost
                            rank_third_elem = map(costMDL,l_solutions, l) #compute costs
                            rank_third_elem = pandas.DataFrame(rank_third_elem, columns = ['cost_MDL']) #convert to DF
                            rank_third_elem.sort_values('cost_MDL', ascending = True, inplace = True)
                            mdl_cost = (rank_third_elem.iloc[0]['cost_MDL']) # take the highest value

                            #store feasible solutions
                            everything_3rd.append(mdl_cost)

                            if mdl_cost < best_cost_3rdElement:
                                print('Cost improved (3rd element) --->', mdl_cost)
                                best_cost_3rdElement = mdl_cost
                                current_cost = [['improvement 3rd element'],[best_cost_3rdElement],[datetime.now() - startTime]]  # to store, track and plot
                                best_arcs = l[rank_third_elem.index[0]].copy()
                                current_arcs = l[rank_third_elem.index[0]].copy()
                                current_solution_mtx = l_solutions[rank_third_elem.index[0]]

                                # also store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                                best_cost_1stElement = costNumberOfRoutes(current_solution_mtx) # already calculated before
                                best_cost_2ndElement =  costStopsDistribution(current_solution_mtx)# already calculated before

                                # reset no_improvement counter
                                no_improvement = 0
                                no_improvement_short_route = 0

                            else:
                                no_improvement += 1 # increase no_improvement counter
                                no_improvement_short_route +=1

                                r = random.random()**beta # beta: bias towards the best neighbor

                                # compare which solutions is the best among the worst
                                degradation_rate_2ndElem = mdl_cost / best_cost_3rdElement
                                degradation_rate_3rdElem = best_cost_2ndElement / stopsDistribution_cost

                                if degradation_rate_2ndElem <= degradation_rate_3rdElem:
                                    r = int(r * len(rank_second_elem))
                                    neighbor_cost = rank_second_elem.iloc[r,0]

                                    delta = best_cost_2ndElement - neighbor_cost

                                    if delta >= 0:
                                        current_arcs = l[rank_second_elem.index[r]].copy()   #find the arcs matrix of the selected neighbor
                                        current_cost = [['accepts worst move because of: equal to zero (2nd element)'],[neighbor_cost],[datetime.now() - startTime]]
                                    elif random.random() <= exp(delta/temperature):
                                        current_arcs = l[rank_second_elem.index[r]].copy()
                                        current_cost = [['accepts worst move because of: greater than random # (2nd element)'],[neighbor_cost],[datetime.now() - startTime]]

                                else:
                                    r = int(r * len(rank_third_elem))
                                    neighbor_cost = rank_third_elem.iloc[r,0]

                                    delta = best_cost_3rdElement - neighbor_cost

                                    if delta >= 0:
                                        current_arcs = l[rank_third_elem.index[r]].copy()   #find the arcs matrix of the selected neighbor
                                        current_cost = [['accepts worst move because of: equal to zero (3rd element)'],[neighbor_cost],[datetime.now() - startTime]]
                                    elif random.random() <= exp(delta/temperature):
                                        current_arcs = l[rank_third_elem.index[r]].copy()
                                        current_cost = [['accepts worst move because of: greater than random # (3rd element)'],[neighbor_cost],[datetime.now() - startTime]]


                else:
                    l = neighborhood(current_arcs, NSize)   #create neighborhood

                    iteration_counter2 += 1

                    l_solutions = []
                    l_feasible = []

                    for i in l:
                        l_solve = solution(i)
                        l_solutions.append(l_solve)
                        l_feas = feasible(l_solve)
                        l_feasible.append(l_feas)

                    # store only feasible solutions
                    l_solutions_new = []
                    l_new = []

                    for i in range(len(l)):
                        if l_feasible[i] == 1:
                            l_solutions_new.append(l_solutions[i])
                            l_new.append(l[i])

                    l_solutions = l_solutions_new # rename
                    l = l_new # rename

                    # IF NO FEASIBLE NEIGHBOR
                    if len(l) == 0:
                        print('(neighborhood empty)')
                        empty_counter+=1
                        continue

                    # compute cost of feasible (1st element) neighbors and rank them

                    rank_first_elem = map(costNumberOfRoutes,l_solutions) #compute costs
                    rank_first_elem = pandas.DataFrame(rank_first_elem, columns = ['cost_NumberOfRoutes']) #convert to DF
                    rank_first_elem.sort_values('cost_NumberOfRoutes', ascending = True, inplace=True) #rank neighbors
                    numberOfRoutes_cost = rank_first_elem.iloc[0]['cost_NumberOfRoutes']

                    everything_1st.append(numberOfRoutes_cost)

                    if numberOfRoutes_cost <  best_cost_1stElement:
                        print('Cost improved (1st element) --->', numberOfRoutes_cost)
                        best_cost_1stElement = numberOfRoutes_cost
                        current_cost = [['improvement 1st element'],[best_cost_1stElement],[datetime.now() - startTime]] # to store, track and plot
                        best_arcs = l[rank_first_elem.index[0]].copy()
                        current_arcs = l[rank_first_elem.index[0]].copy()
                        current_solution_mtx = l_solutions[rank_first_elem.index[0]]

                        # also store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                        best_cost_2ndElement = costStopsDistribution(current_solution_mtx)
                        best_cost_3rdElement = costMDL(current_solution_mtx,best_arcs)

                        # reset no_improvement counter
                        no_improvement = 0
                        iteration_counter2 += 1

                    else: # else compute second element cost
                        rank_second_elem = map(costStopsDistribution,l_solutions) #compute costs
                        rank_second_elem = pandas.DataFrame(rank_second_elem, columns = ['cost_stopsDistribution']) #convert to DF
                        rank_second_elem.sort_values('cost_stopsDistribution', ascending = False, inplace = True)
                        stopsDistribution_cost = (rank_second_elem.iloc[0]['cost_stopsDistribution']) # take the highest value

                        everything_2nd.append(stopsDistribution_cost)

                        if stopsDistribution_cost < best_cost_2ndElement:
                            print('Cost improved (2nd element) --->', stopsDistribution_cost)
                            best_cost_2ndElement = stopsDistribution_cost
                            current_cost = [['improvement 2nd element'],[best_cost_2ndElement],[datetime.now() - startTime]] # to store, track and plot
                            best_arcs = l[rank_second_elem.index[0]].copy()
                            current_arcs = l[rank_second_elem.index[0]].copy()
                            current_solution_mtx = l_solutions[rank_second_elem.index[0]]

                            # also compute and store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                            best_cost_1stElement = costNumberOfRoutes(current_solution_mtx) # already calculated before
                            best_cost_3rdElement = costMDL(current_solution_mtx,best_arcs)

                            # reset no_improvement counter
                            no_improvement = 0
                            iteration_counter2 += 1

                        else: # compute 3rd element cost
                            rank_third_elem = map(costMDL,l_solutions, l) #compute costs
                            rank_third_elem = pandas.DataFrame(rank_third_elem, columns = ['cost_MDL']) #convert to DF
                            rank_third_elem.sort_values('cost_MDL', ascending = True, inplace = True)
                            mdl_cost = (rank_third_elem.iloc[0]['cost_MDL']) # take the highest value

                            everything_3rd.append(mdl_cost)

                            if mdl_cost < best_cost_3rdElement:
                                print('Cost improved (3rd element) --->', mdl_cost)
                                best_cost_3rdElement = mdl_cost
                                current_cost = [['improvement 3rd element'],[best_cost_3rdElement],[datetime.now() - startTime]]  # to store, track and plot
                                best_arcs = l[rank_third_elem.index[0]].copy()
                                current_arcs = l[rank_third_elem.index[0]].copy()
                                current_solution_mtx = l_solutions[rank_third_elem.index[0]]

                                # also store results of 2nd and 3rd element (cost) in order to properly compare in the next iteration
                                best_cost_1stElement = costNumberOfRoutes(current_solution_mtx) # already calculated before
                                best_cost_2ndElement =  costStopsDistribution(current_solution_mtx)# already calculated before

                                # reset no_improvement counter
                                no_improvement = 0
                                iteration_counter2 += 1

                            else:
                                no_improvement += 1 # increase no_improvement counter

                                r = random.random()**beta # beta: bias towards the best neighbor

                                # compare which solutions is the best among the worst
                                degradation_rate_2ndElem = mdl_cost / best_cost_3rdElement
                                degradation_rate_3rdElem = best_cost_2ndElement / stopsDistribution_cost

                                if degradation_rate_2ndElem <= degradation_rate_3rdElem:
                                    r = int(r * len(rank_second_elem))
                                    neighbor_cost = rank_second_elem.iloc[r,0]

                                    delta = best_cost_2ndElement - neighbor_cost

                                    if delta >= 0:
                                        current_arcs = l[rank_second_elem.index[r]].copy()   #find the arcs matrix of the selected neighbor
                                        current_cost = [['accepts worst move because of: equal to zero (2nd element)'],[neighbor_cost],[datetime.now() - startTime]]
                                    elif random.random() <= exp(delta/temperature):
                                        current_arcs = l[rank_second_elem.index[r]].copy()
                                        current_cost = [['accepts worst move because of: greater than random # (2nd element)'],[neighbor_cost],[datetime.now() - startTime]]

                                else:
                                    r = int(r * len(rank_third_elem))
                                    neighbor_cost = rank_third_elem.iloc[r,0]

                                    delta = best_cost_3rdElement - neighbor_cost

                                    if delta >= 0:
                                        current_arcs = l[rank_third_elem.index[r]].copy()   #find the arcs matrix of the selected neighbor
                                        current_cost = [['accepts worst move because of: equal to zero (3rd element)'],[neighbor_cost],[datetime.now() - startTime]]
                                    elif random.random() <= exp(delta/temperature):
                                        current_arcs = l[rank_third_elem.index[r]].copy()
                                        current_cost = [['accepts worst move because of: greater than random # (3rd element)'],[neighbor_cost],[datetime.now() - startTime]]

                costs_achieved.append(current_cost)

            except Exception as e:
                print(e)
                print('#')
                pass

        temperature = temperature * alpha

    #plt.plot(costs_achieved)

    return best_arcs


"""
LARGE NEIGHBORHOOD SEARCH ALGORITHM (LNS)
"""


def relatedness(sol_df1, i,j):
    global wage, dist_mtx

    if int(sol_df1.loc[i,'route'] == sol_df1.loc[j,'route']): v = 0 # average distance between points in distance matrix
    else: v = 121292

    c = (dist_mtx.iloc[i,j])

    return 1/(c+v)


def RemoveSet(base_list,removing_list):

    A = copy.copy(base_list)

    for i in removing_list:
        A.remove(i)

    return A

def SelectCustomers(solution_df, n):

    S1 = [random.choice(customers)]

    for i in range(2,n+1): # n starts in 1
        C = random.choice(S1)
        nonS1 = RemoveSet(customers,S1) #remove S from customers
        relatedness_ranking = sorted(nonS1, key = functools.partial(relatedness, solution_df, C), reverse = True) #sort (descending order) nonS1 customers according to relatedness to C
        rank_number = int(random.random() ** beta * len(nonS1))  # Select a biased-random related node
        next_node = relatedness_ranking[rank_number]
        S1.append(next_node)

    return S1


def neighborhood2(arcs_mtx,S,size):

    for retry in range(1): # if no feasible neighborhood is found, try again
        neighbors = list()
        nonS = RemoveSet(customers,S)
        ns = min(size,len(nonS)-1) # corrected neighborhood size
        arcs_neighbor = arcs_mtx.copy()

        for iter in range(ns):
            relocated = arcs_neighbor.copy()
            print('... relocating')
            for from_node in S:
                to_node = random.choice(nonS)
                relocated = relocate(relocated,from_node,to_node).copy()

                sol1 = solution(relocated)

                if feasible(sol1) == 1:
                    print('.')
                    arcs_neighbor = relocated.copy()
                    neighbors.append(arcs_neighbor)

        if len(neighbors) > 0: break

    return neighbors

def costDirectFromArcs_NRoutes(arcs_mtx):
    return costNumberOfRoutes(solution(arcs_mtx))

def costDirectFromArcs_OF(arcs_mtx):
    return objectiveFunction(solution(arcs_mtx))

def insertion(arcs_mtx, c, nodes_pair): # c = customer to be inserted

    arcs_new = arcs_mtx.copy()

    c_succ = successor(arcs_new, c)
    c_pred = predecessor(arcs_new, c)

    node_a = nodes_pair[0]
    node_b = nodes_pair[1]

    # delete edge between c and its predecessor and c and its succesor
    arcs_new.loc[c_pred, c] = 0
    arcs_new.loc[c, c_pred] = 0

    arcs_new.loc[c_succ, c] = 0
    arcs_new.loc[c, c_succ] = 0

    # create edge between c's succesor and predecessor
    arcs_new.loc[c_pred, c_succ] = 1
    arcs_new.loc[c_succ, c_pred] = 1

    # delete edge between node a and node b
    arcs_new.loc[node_a, node_b] = 0
    arcs_new.loc[node_b, node_a] = 0

    # create edge between insertion_location and the customer to be inserted
    arcs_new.loc[node_a, c] = 1
    arcs_new.loc[c, node_a] = 1

    arcs_new.loc[node_b, c] = 1
    arcs_new.loc[c, node_b] = 1

    columnsList = list(arcs_new.columns)

    position_c = columnsList.index(c)
    position_node_a = columnsList.index(node_a)

    if position_c < position_node_a:
        change = list(arcs_new[c]) # store column i in a list apart
        arcs_new = arcs_new.drop(c, 1) # delete column i from df (1 indicates to drop column and not row)
        arcs_new.insert(position_node_a, column = c, value = change) # relocate column i

    if position_c > position_node_a:
        change = list(arcs_new[c]) # store column i in a list apart
        arcs_new = arcs_new.drop(c, 1) # delete column i from df
        arcs_new.insert(position_node_a+1, column = c, value = change) # relocate column i (this changes compared to the above case)

    return arcs_new.astype(int)


def insertS(arcs_mtx, S, pairs_list): # S: set of customers S, pairs_list: list of edges of the route where S should be inserted

    arcs_c_S = arcs_mtx.copy()
    max_retry = 2

    neighbors = []
    for i in range(max_retry):
        for s in S:
            insertion_location = random.choice(pairs_list)
            if insertion_location[0] != s and insertion_location[1] != s : # make sure that s is different to the random location
                # insert s
                ins_s = insertion(arcs_c_S,s,insertion_location)
                neighbors.append(ins_s)

    return neighbors


def LNS(arcs_mtx):

    Initialize()

    best_arcs2 = arcs_mtx.copy()

    current_plan = best_arcs2

    best_solution2 = solution(best_arcs2)

    best_cost2 = objectiveFunction(best_solution2)

    costs_achieved2 = [best_cost2]

    no_improvement = 0

    everything = [] # to store all feasible solutions found

    print('\n- Begin LNS', '\nstarting_cost =', best_cost2)

    for iter1 in range(maxiter1):
        for N in range(1,p+1): # number of customers allowed to perform the search
            for iter2 in range(maxiter2):
                if no_improvement >= max_no_improvement:
                    print('no improvement!')
                    no_improvement = 0
                    break

                print('Iter 3rd level=', iter1, 'Iter No of Customers (N) =', N,'  Iter first level=',iter2)

                S = SelectCustomers(solution(best_arcs2), N)


                S_inserted = []
                for i in range(sizeLNS):

                    current_sol = best_solution2.copy()

                    # find edges of the route where S should be inserted
                    s = random.choice(S) # choose a random s of S (for choosing randomly a route where to insert S)

                    # find route number of s
                    s_route = current_sol.loc[s, 'route']

                    # select information only from route s
                    current_sol_sorted = current_sol.loc[(current_sol.loc[:,'route'] == s_route) | (current_sol.loc[:,'route'] == store_id)] # filter (include depot info)
                    current_sol_sorted.loc[0,'stop_number'] = 0 # replace NA of depot
                    current_sol_sorted = current_sol_sorted.sort_values('stop_number', ascending = True) # sort

                    customers_list = list(current_sol_sorted.index)

                    # a nodes
                    a_nodes = customers_list
                    # b nodes
                    b_nodes = a_nodes[1:] #select from 2nd element on
                    b_nodes.append(store_id2) # add las stop at depot
                    # create pair of nodes a and b
                    pairs_a_b = pandas.DataFrame( {'a': pandas.Series(a_nodes),
                                              'b': pandas.Series(b_nodes)})    # transform to list of pairs
                    pairs_a_b = pairs_a_b.values.tolist()

                    # insert S
                    insert_S = insertS(best_arcs2,S,pairs_a_b)

                    # store results
                    S_inserted = S_inserted + insert_S

                # check feasibility
                s_solutions = []
                s_feasible = []
                for i in S_inserted:
                    s_solve = solution(i)
                    s_solutions.append(s_solve)
                    s_feas = feasible(s_solve)
                    s_feasible.append(s_feas)

                # store only feasible solutions
                s_solutions_new = []
                s_new = []

                for i in range(len(S_inserted)):
                    if s_feasible[i] == 1:
                        s_solutions_new.append(s_solutions[i])
                        s_new.append(S_inserted[i])


                s_solutions = s_solutions_new # rename
                s_plans = s_new # rename

                # If no feasible neighbor
                if len(s_plans) == 0:
                    print('neighborhood empty')
                    continue

                # compute cost of feasible neighbors and rank them
                rank = map(objectiveFunction,s_solutions) #compute costs
                rank = pandas.DataFrame(rank, columns = ['cost_objectiveFunction']) #convert to DF
                rank.sort_values('cost_objectiveFunction', ascending = True, inplace=True) #rank neighbors

                # obtain cheapest cost, plan, and solution
                cheapest_cost = rank.iloc[0,0]
                everything.append(cheapest_cost)

                if cheapest_cost < best_cost2:
                    print('Cost improved:', cheapest_cost)

                    percent_improv = (best_cost2 - cheapest_cost)/best_cost2

                    best_cost2 = cheapest_cost
                    best_arcs2 = s_plans[rank.index[0]]
                    best_solution2 = s_solutions[rank.index[0]]

                    costs_achieved2.append(best_cost2)

                    if percent_improv < 0.1 :
                        no_improvement += 1 # count if there is less than 1% improvement
                        print('improvement less than 1% no_improvement # =', no_improvement)
                    else: no_improvement = 0

                else: no_improvement += 1

                print('no_improvement =',no_improvement)

    print(datetime.now() - startTime)

    #plt.plot(costs_achieved2)

    return best_arcs2




