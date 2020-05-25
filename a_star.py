# -*- coding: utf-8 -*-

import numpy as np

def h(bla):
    """
    HEURISTIC FUNCTION FOR A*
    """
    return 0

def csv_matr (filename):
    """
    ALLOWS IMPORTING CONNECTIVITY MATRIX FROM CSV
    """
    import csv

    results = np.array([])
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        results = np.array(list(map(np.asarray,reader)))
    
    results = mirrorize(results).astype(int)    
    return results

def matr2listOfEdges (CONN):
    
    results = [];
    for i in np.arange(np.shape(CONN)[0]):
        for j in np.arange(np.shape(CONN)[1]):
            if CONN[i,j] > 0:
                results.append((i,j))
    return results
    
    #G.add_edges_from([(1, 2), (1, 3)])

def mirrorize (CONN):
    """
    RESTORES SYMMETRICAL VALUES IN EVERY MATRIX CONNECTION
    """
    CONN_t = CONN.transpose()
    CONN_fin = (CONN + CONN_t) > 0
    return CONN_fin
    

def reconstruct_path(cameFrom, current):
    """
    RETURNS PATH ARRAY
    """
    total_path = np.array([current])
    while current != -1:
        current = cameFrom[int(current)]
        total_path = np.insert(total_path, 0, current)
    return total_path[1:]

def neighbours (CONN, satno):
    """
    FINDS NEIGHBOURS OF satno IN CONN MATRIX
    """
    neig = np.array([])
    
    for i in np.arange(np.shape(CONN)[0]):
        if CONN[satno,i] != 0:
            neig = np.append(neig,i)
            
    return neig
            
def minopenfscore( openSet, fScore ):
    """
    LOCATES A NODE IN openSet THAT HAS THE LOWEST fScore
    """
    L = np.size(fScore);
    
    for q in np.arange(L):
        if not np.any(openSet==q):
            fScore[q] = np.inf
            
    mini = np.argmin(fScore)
    return mini



# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    
def A_Star(start, goal, CONN, h):
    
    Nsat = np.shape(CONN)[0]
    
    openSet = np.array([start])
    
    closedSet = np.array([])

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = np.zeros(Nsat) - 1

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = np.zeros(Nsat) + np.inf
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n).
    fScore = np.zeros(Nsat) + np.inf
    fScore[start] = h(start)

    while openSet.size > 0:
        
        #print(openSet)
        #print(closedSet)
        
        current = minopenfscore(openSet, fScore)
        
        #print(current)
        
        #print(current)
        #print(reconstruct_path(cameFrom, current))
        
        if current == goal:
            return reconstruct_path(cameFrom, current)

        openSet = openSet[openSet != current]
        closedSet = np.append(closedSet, current)
        closedSet.sort()
        
        for neighbour in neighbours(CONN, current):
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            
            i_n = int(neighbour) 
            
            if np.any(closedSet == neighbour):
                continue
            
            tentative_gScore = gScore[current] + 1 #1 is supposed to be d(current, neighbor) - the weight, but fuck it for now
            
            if not np.any(openSet == neighbour):
                openSet = np.append(openSet, neighbour)
                openSet.sort()
            else:
                if tentative_gScore >= gScore[i_n]:
                    continue
            
            cameFrom[i_n] = current
            gScore[i_n] = tentative_gScore
            fScore[i_n] = gScore[i_n] + h(i_n)
            

    # Open set is empty but goal was never reached
    return 0