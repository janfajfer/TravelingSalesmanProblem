#!/usr/bin/env python3
"""
A Traveling Salesman Problem solver applied on shredded images.

Part of the assignment for Combinatorial optimization class at FEE CTU.
Sources:
    - https://cw.fel.cvut.cz/b192/_media/courses/ko/10_tsp.pdf

Usage:
    main.py [input file] [output file]
"""

# imports ----------------------------------------
import gurobipy as grb
import sys
import numpy as np


def compute_distance(i: int, j: int, S: np.array):
    """
    Computes distances for two slices of an image.
    :param i: i-th slice of the image
    :param j: j-th slice of the image
    :param S: matrix representing the shredded image
    :return: distance between the two slices
    """
    width_index = S.shape[2] - 1
    s1 = S[i, :, width_index, :]
    s2 = S[j, :, 0, :]
    return np.sum(np.abs(s1 - s2))

def find_smallest_subcycle(start, outgoing: np.array, closed: np.array):
    """
    For an array representing nodes and outgoing edges finds the smallest sub-cycle.

    :param start: starting node index
    :param outgoing: array specifying an outgoing edge from a node
    :param closed: nodes that are closed
    :return: (subcycle_length, subcycle): length and the subcycle itself
    """
    closed[start]=True
    current=outgoing[start]
    subcycle = [start]
    subcycle_length = 1

    # does a loop over the graph and ends when reaches the starting index
    while current != start:
        closed[current] = True
        subcycle.append(current)
        subcycle_length=subcycle_length+1
        current=outgoing[current]

    subcycle.append(start)
    return subcycle_length, subcycle

def my_callback(model, where):
    # Callback is called when an event occurs.
    # An integer solution event corresponds to GRB.Callback.MIPSOL

    if where == grb.GRB.Callback.MIPSOL:
        # if an integer solution is found
        # get the edges included in the solution
        vars = model.getVars()
        values = model.cbGetSolution(model.getVars())
        outgoing = np.ones(model._n, dtype=int) * -1
        closed = np.zeros(model._n, dtype=bool)
        for i in range(len(values)):
            if values[i] > 0:
                from_i, to_j = [int(num) for num in vars[i].VarName.split(':')]
                outgoing[from_i] = to_j

        # go through all cycles and find the smallest one
        smallest_subcycle_length = model._n
        smallest_subcycle = []
        for i in range(len(outgoing)):
            if not closed[i]:
                subcycle_length, subcycle = find_smallest_subcycle(i, outgoing, closed)
                if subcycle_length < smallest_subcycle_length:
                    smallest_subcycle_length = subcycle_length
                    smallest_subcycle = subcycle

        # add a constraint for the smallest cycle if needed
        if smallest_subcycle_length != model._n:
            # all the variable in the smallest subcycle should sum to smallest_subcycle_length -1
            smallest_subcycle_variables = []
            for i in range(smallest_subcycle_length):
                smallest_subcycle_variables.append(model.getVarByName(str(smallest_subcycle[i]) + ':' + str(smallest_subcycle[i+1])))
            model.cbLazy(grb.quicksum(var for var in smallest_subcycle_variables) <= smallest_subcycle_length-1)


def solveTSP(distances: list):
    # Formulating ILP
    # model -----------------------------------------------------
    model = grb.Model()
    model.Params.lazyConstraints = 1

    # - ADD VARIABLES
    model._n = n + 1
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(vtype=grb.GRB.BINARY, name=f'{i}:{j}', obj=distances[i][j])
    # dummy node
    for i in range(n):
        x[i, n] = model.addVar(vtype=grb.GRB.BINARY, name=f'{i}:{n}', obj=0)
        x[n, i] = model.addVar(vtype=grb.GRB.BINARY, name=f'{n}:{i}', obj=0)

    # - ADD CONSTRAINTS
    # only one outgoing and one incoming edge to a node
    for i in range(n + 1):
        model.addConstr(grb.quicksum(x[i, j] for j in range(n+1) if j != i) == 1)
        model.addConstr(grb.quicksum(x[k, i] for k in range(n+1) if k != i) == 1)

    model.optimize(my_callback)

    print(f"objective value: {model.getObjective().getValue()}")

    # retrieve the optimal solution
    vars = model.getVars()
    outgoing = np.ones(model._n, dtype=int) * -1
    for v in vars:
        if v.X > 0:
            from_i, to_j = [int(num) for num in v.VarName.split(':')]
            outgoing[from_i] = to_j

    # get optimal circuit
    current=outgoing[n]
    optimal_circuit = []
    while current != n:
        optimal_circuit.append(current)
        current=outgoing[current]

    return optimal_circuit

# Input ----------------------------------------

with open(sys.argv[1]) as f:
    n, w, h = [int(x) for x in f.readline().split()]
    S = np.zeros(shape=[n, h, w, 3])
    for i in range(n):
        S[i] = np.array([int(i) for i in f.readline().split()]).reshape([h, w, 3], order='C')

# Compute distances for all slices of the input image
distances = np.zeros((n,n),dtype=int)
for i in range(n):
    for j in range(n):
        if i != j:
            distances[i][j]=compute_distance(i, j, S)
optimal_circuit = solveTSP(distances)

# Write the output to a file ----------------------------------------
optimal_circuit_str = " ".join(str(num + 1) for num in optimal_circuit)
print(optimal_circuit_str)
with open(sys.argv[2], "w") as f:
    f.write(optimal_circuit_str + "\n")
