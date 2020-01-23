from __future__ import print_function
from functools import partial
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import numpy as np

#Import data

#rawData = # import data as a dataframe
#distanceMatrix = # import data as a dataframe
#timeMatrix = # import data as a dataframe


###########################
# Problem Data Definition #
###########################
def create_data_model():
    """Stores the data for the problem"""
    data = {}

    data['distance_matrix'] = distanceMatrixList

    data['time_matrix'] = timeMatrixList

    data['num_locations'] = 1133

    data['time_windows'] = timeWindowsList

    data['demands'] = list(rawData["Demand"])

    data['time_per_demand_unit'] = 5  # 5 minutes/unit

    data['num_vehicles'] = 50

    data['vehicle_capacity'] = [900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000, 900000000,
                                9000000000,	9000000000,	9000000000,	9000000000,	9000000000,	9000000000,	9000000000,	9000000000,	9000000000,	9000000000,
                                12000000000, 12000000000, 12000000000, 12000000000,	12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000, 12000000000,
                                82000000000, 82000000000, 82000000000,
                                47000000000, 47000000000, 47000000000]
    data['depot'] = 0
    return data


###########
# Printer #
###########
def print_solution(data, manager, routing, assignment):  # pylint:disable=too-many-locals
    """Prints assignment on console"""

    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    print(dropped_nodes)

    # Print the rest of the solution
    print('Objective: {}'.format(assignment.ObjectiveValue()))
    total_distance = 0
    total_load = 0
    total_time = 0
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    time_dimension = routing.GetDimensionOrDie('Time')
    for vehicle_id in xrange(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)
            slack_var = time_dimension.SlackVar(index)
            plan_output += ' {0} Load({1}) Time({2},{3}) Slack({4},{5}) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var),
                assignment.Min(time_var),
                assignment.Max(time_var),
                assignment.Min(slack_var), assignment.Max(slack_var))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     vehicle_id)

        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += ' {0} Load({1}) Time({2},{3})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var),
            assignment.Min(time_var), assignment.Max(time_var),
            assignment.Min(slack_var), assignment.Max(slack_var))

        plan_output += 'Distance of the route: {0}m\n'.format(distance)
        plan_output += 'Load of the route: {}\n'.format(
            assignment.Value(load_var))
        plan_output += 'Time of the route: {}\n'.format(
            assignment.Value(time_var))
        print(plan_output)
        total_distance += distance
        total_load += assignment.Value(load_var)
        total_time += assignment.Value(time_var)

    print('Total Distance of all routes: {0}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))
    print('Total Time of all routes: {0}min'.format(total_time))


########
# Main #
########
def main():
    """Entry point of the program"""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacity'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Allow to drop nodes.
    penalty = 4967922279
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Add Time Window constraint
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        1109,  # allow waiting time 990
        1109,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))

    for vehicle_id in range(data['num_vehicles']):
        time_dimension.SetSpanUpperBoundForVehicle(int(550), vehicle_id)


    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)  #

    # Setting the metaheuristic
    #search_parameters.local_search_metaheuristic = (
    #routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    #search_parameters.time_limit.seconds = 10800
    #search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()

