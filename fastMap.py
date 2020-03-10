#!/usr/bin/env python3

# This python implementation of the FAST MAP algorith has been done as pt_a joint work of Arpit Parwal (aparwal@usc.edu) and Yeon-soo Park ( )
# The distances are provided in pt_a separate data file


# Imports
from collections import namedtuple
import copy
import numpy as np
# for matlab 
import matplotlib.pyplot as plt
# End imports


# function to calculate the farthest distance pt_b/w any two points
def getFarthestPoints(distances):
    first = np.random.randint(0, 9)
    while True:
        second = np.argmax(distances[first])
        tmp = np.argmax(distances[second])
        if tmp == first:
            break
        else:
            first = second

    res = (min(first, second), max(first, second))
    return res

 
    

# recomputing new distances based on previous distanc
def computeNewDistances(oldDistances, lastDims):
    res = copy.deepcopy(oldDistances)
    num_points = len(oldDistances)
    for i in range(num_points):
        for j in range(num_points):
            tmp = oldDistances[i][j]**2 - (lastDims[i] - lastDims[j])**2
            res[i][j] = np.sqrt(tmp)
    return res

# main driver function for fast Map
def run_fastMap(distances, dimension):

    # 
    res = [[] for _ in distances]
    # for number of dimensions O(K) times
    for _ in range(dimension): 
        # to get two farthest point //Constant time
        first, second = getFarthestPoints(distances)
        # calculate distance //constant point
        farthestDist = distances[first][second]

        # In order to compute for each point of the array // O(N) times 
        for pt in range(len(distances)):
            # edge case 
            if pt == first:
                dist = 0
            # edge case 2
            elif pt == second:
                dist = farthestDist
            else:
                #  to calulate trianguated position
                dist = (distances[first][pt]**2 + farthestDist**2 - distances[second][pt]**2)/(2 * farthestDist)                 

            res[pt].append(dist)
        # updating the distance function 
        distances = computeNewDistances(
            distances, list(map(lambda x: x[-1], res)))

    return res


if __name__ == '__main__':
    # path for input file
    input_file = './data/fastmap-data.txt'
    # count of number of points
    num_points=int(10)
    # the value for dimensions
    dimension=int(2)
    # path to the word file
    word_file = './data/fastmap-wordlist.txt'

    # array to store data from the data file
    # this helps in avoiding mltiple fetches to the data base
    data = []

    # getting data from the file and make it beautiful so that we can use it for our purpose
    with open(input_file) as fh:
        for line in fh:
            pt_a, pt_b, dist = line.split()
            data.append([int(pt_a), int(pt_b), float(dist)])
    distances = np.zeros((num_points,num_points))
    for pt_a, pt_b, dist in data:
        distances[pt_a-1][pt_b-1], distances[pt_b-1][pt_a-1] = dist, dist

    coOrds = run_fastMap(distances, dimension)


    # print (distances)
    # print(data)
    
    #  This code is to print and plot data into a 2d graph for better visualisation.
    wordList = []
    with open(word_file) as fh:
        wordList = list(map(str.strip, fh.readlines()))

    for word, coOrd in zip(wordList, coOrds):
        print('Cordinate for  {0} - {1}'.format(word, coOrd))
    # for plotting the result on a graph to show visually 
    if dimension == 2:
        wordList = []
        with open(word_file) as fh:
            wordList = list(map(str.strip, fh.readlines()))
        coOrds = np.asarray(coOrds)
        fig, ax = plt.subplots()
        ax.scatter(coOrds[:, 0], coOrds[:, 1])
        for idx, label in enumerate(wordList):
            ax.annotate(label, (coOrds[idx]))
        plt.show()
    else:
        print('Can only plot 2D !')
