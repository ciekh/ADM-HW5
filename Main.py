import networkx as nx
from collections import defaultdict
import heapq as hp
import folium
from folium import plugins
import ipywidgets
from tkinter import *
from tkinter import messagebox


#Graph Network ---------------------------
G = nx.Graph()
#NODES of GRAPH Network

with open(r'USA-road-d.CAL.co','r') as f1:
    for line in f1:
        if line[0] == 'v':
            n,lo,la = list(map(int, line.strip().split()[1:]))
            G.add_node(n,latitude = la/1000000,longitude = lo/1000000)

#Edges (Distance) of Graph Network

adj = defaultdict(set)
with open(r'USA-road-d.CAL.gr','r') as f:
    for line in f:
        if line[0] == 'a':
            n1,n2, d =  list(map(int, line.strip().split()[1:]))
            G.add_edge(n1,n2,distance = d,weight = 1)
            adj[n1].add(n2)
            adj[n2].add(n1)


#Edges (Time) of Graph Network

with open(r'USA-road-t.CAL.gr','r') as f2:
    for line in f2:
        if line[0] == 'a':
            n1,n2, t =  list(map(int, line.strip().split()[1:]))
            G.add_edge(n1,n2,time = t)



def neighbors(v,w,d,graph = G, adjacent = adj):
    
    # In the visited dictionary I insert node and minimum weight to reach it.
    # Usually with the set of nodes visited in the djikstra algorithm we mean all the nodes that are visited 
    # and for which the weight is no longer changed but here I include all the ones that can be changed 
    # but at the end of the while cycle at each node will be assigned the final weight
    visited = {v: 0}
    F = []
    # F is a heap where I keep all the nodes that can extend the tree of the shortest paths, 
    # I use a minheap because I have to extract the minimum and the extraction of the minimum happens in O (1) time
    # even if then reassigning to the root the minimum value uses O (logn) time is reasonable.
    hp.heapify(F)
    # the input node v is the starting node
    current_node = v
    edges_min_d = set()
    edges_max_d = set()
    while True:
        # I take all the nodes adjacent to v
        adjacents = adjacent[current_node]
        # I take the weight of the current node from the visited dictionary
        weight_to_current_node = visited[current_node]
        for node in adjacents:
            # I take the weights of all the adjacent ones calculating it as the sum of the weight of the adjacent node 
            # and the weight of the edge that connects them according to the weight function passed in input.
            weight = graph[current_node][node][w] + weight_to_current_node
            #here I filter the data, I take only those at a distance less than d.
            #I insert only the nodes that can extend the tree of the shortest paths but weigh less than the treshold d
            #in visited and in the border F.
            if weight > d:
                edges_max_d.add((current_node,node))
            else:
                edges_min_d.add((current_node,node))
                # if the node is the first time I reach it simply add to the visited and at the border 
                #the node with its weight calculated as the sum of the weight of the node that allowed it to be reached 
                #and the weight of the arch that connects them
                if node not in visited:
                    visited[node] = weight
                    hp.heappush(F,(weight,node))
                else:
                    #instead if the node has already been visited and the distance assigned to the node
                    #is greater than that given by the sum of the weight of the new node that allowed us to reach 
                    #it and the edge that connects them then the weight associated with the node is updated
                    #with this last.And a new weight node tuple is inserted into the heap.
                    #Here is the only weakness of our algorithm (at least the ones I saw)
                    #inasmuch as having more weight node pairs it will be more iterations than necessary, 
                    #this however does not impact on the correctness of the result because I from the heap simply
                    #take the node while I recover the weight from the visited dictionary where it is updated
                    #with the best value, at time level a little slows down to do more iterations but a reasonable 
                    #time having done many tests I also tried to eliminate this problem but the only way 
                    #I found was to go over a list to delete the tuple and insert the new tuple to 
                    #then recreate the heap but doing various tests these operations were heavier than doing 
                    #some more iteration.
                    current_shortest_weight = visited[node]
                    if current_shortest_weight > weight:
                        visited[node] = weight
                        hp.heappush(F,(weight,node))
        # I extract the vertices until the border is empty, this means that in my dictionary 
        #I will have all the nodes at a distance less than d, in fact they have the list of keys of visited
        try:
            current_node = hp.heappop(F)[1]
        except:
            break
    neighbors = list(visited.keys())
    edges_max_d = list(edges_max_d)
    edges_min_d = list(edges_min_d)
    
    maxres = edges_max_d
    maxlst = []
    for y in range (len(maxres)):
        maxlst.append(maxres[y][0])
        maxlst.append(maxres[y][1])

    notneighbors = list(set(maxlst) - set(neighbors))
    
    return (neighbors,edges_min_d,edges_max_d,notneighbors)

#Function 2 

#new
#n_trees keeps the count of recursive calls of the function, and gives the number of different trees built
def function2(nodes, d, n_trees = 1):
    # First we check if all nodes have at least one edge conecting them to one of the other nodes passed in input.
    # Lonely notes are stored in a list
    loners = set()
    for n in nodes:
        if adj[n].intersection(nodes) == set():
            loners.add(n)
    nodes = nodes - loners 
    # Now we are sure that the nodes for which we want to find networks are at least connected to another node
    # of the input set. This means that we may find more than one tree, because there might still be unconnected
    # groups of nodes, that will form separated networks. Instead of finding one minimum spanning tree, here we look
    # for a forest of minimum spanning trees that we can create from any set of nodes passed in input
    
    # We store all edges and edges' lenghts in a heap structure, so we can then take the global minimum edge.
    edges = []
    hp.heapify(edges)
    newset = nodes.copy()
    for i in nodes:
        #newset is necessary in order to take only once each minimal path, so we don't take the same edges twice
        newset.remove(i)
        for j in newset:
            if j in adj[i]:
                edge = [i,j]
                length = G[i][j][d]
                hp.heappush(edges, (length, edge))
                
    #The algorithm starts taking the minimum edge between all edges that connect nodes from the input, and storing
    #a set of visited nodes starting with the first two, and a set of edges starting with the first minimum edge
    visited = set()
    new_edges = edges.copy()
    e = new_edges.pop(0)[1]
    visited = visited.union(set(e))
    out = {tuple(e)}
    
    #This condition checks if we already reached all the connected nodes (in one of the recursions), at the first step
    if visited == nodes:
            max_trees = n_trees
    while visited != nodes:
        #the loop goes on until all nodes given in input are visited, and takes always the minimum of the 
        #remaining edges that connects only one of the visited nodes to one of the not visited; this way, we can
        #be sure that we are connecting these minimal paths to the same network, without forming cycles
        e = new_edges.pop(0)[1]
        if len(set(e).intersection(visited)) == 1:
            out = out.union({tuple(e)})
            visited = visited.union(set(e))
            new_edges = edges.copy()
            
        # The following condition is used to to check if there's more than one connected group of nodes. If that's the
        # case, so we have checked all of the paths connected to the starting edge, but we didn't manage to visit 
        # all the nodes, than we reapply the same function on the subset of nodes not yet visited, and by recursion
        # build all possible trees if there's more than one, until we visit all the nodes passed in input
        # (minus the loners, that we already took away from nodes)
        if len(new_edges) == 0:
            new_tree = function2(nodes-visited, d, n_trees = n_trees +1)
            out = out.union(new_tree)
            visited = visited.union({node for tup in new_tree for node in tup})
        
        #This will be updated only in the recursion step in which the function doesn't need recursion, so the final
        #step, and if we reach this condition we can store the number of total trees built through the whole function
        elif visited == nodes:
            max_trees = n_trees
    # The output is the set of edges that connect all the nodes of each tree (not all edges  will belong to the same 
    # tree). These are the edges that make possible to visit all (connected) nodes with minimum cost, starting from
    # any other node ( if connected to the same tree).
    if loners != set():    
        print('In the set of nodes given there were {} unconnected nodes!\n'.format(len(loners)))
    
    #Try... except is needed in order to skip the recursive steps in which we have not yet visited all the nodes,
    #so we don't have the total number of trees yet
    try:
        print('Total number of trees built was {}!'.format(max_trees))
    except:
        pass
    return out



#Function 2 Visualization funct

#new
# if no weight is inserted in the query, the default weight will be the network distance
#new
# if no weight is inserted in the query, the default weight will be the network distance
def fun2_visual(subset, w = 'weight'):
    
    if len(subset) <= 1000:
        
        try:
            edges = function2(subset, w)
            
            # just taking the first node from the edges set, in order to locate the map on one of the trees
            map_loc = next(iter(edges))[0]
        except:
            return 'No connection found between points given in input'

        coor = G.nodes[map_loc]
        ourmap = folium.Map(location=[coor['latitude'],coor['longitude']], zoom_start=8)
        

        #This chunk of code is used to represent also unconnected points, but folium doesn't support
        #large number of markers. If the set of nodes in input is small, this can be executed, otherwise
        #we only represent the trees (edges). Also, for large sets of nodes, there would be many unconnected nodes
        #and the visualiation would be too confused

        if len(subset) <= 500:
            for node in subset:
                coor = G.nodes[node]
                folium.Marker(location=[coor['latitude'],coor['longitude']], popup = str(node), 
                              icon=folium.Icon(color='lightblue')).add_to(ourmap)
        else:
            print('Visualizing only connected nodes, given high number of unconnected nodes:')

        for edge in edges:
            points = []
            for node in edge:
                coor = G.nodes[node]
                folium.Marker(location=[coor['latitude'],coor['longitude']], popup = str(node),
                              icon=folium.Icon(color='blue')).add_to(ourmap)

                points.append((coor['latitude'],coor['longitude']))
            if w == 'distance':
                folium.vector_layers.PolyLine(points, color = 'Red').add_to(ourmap)            
            elif w == 'time':
                folium.vector_layers.PolyLine(points, color = 'Green').add_to(ourmap)
            elif w == 'weight':
                folium.vector_layers.PolyLine(points, color = 'Purple').add_to(ourmap)
        return ourmap
    else:
        return 'Too many nodes given in input, visualization not possible!'


def dijkstra_modified(v,end,p,graph = G,adjacent = adj):
    # v is the start node
    # end is the destination node
    # Also here as in the functionality 1 I use djikstra with small changes here instead of calculating 
    # the neighbors of the starting node I calculate the minimum path between the starting node and the arrival node.
    # I create a dictionary of the visited ones but besides the weight I save the predecessor
    # that then it will be useful to me to reconstruct the path
    visited = {v: (None, 0)}
    F = []
    hp.heapify(F)
    current_node = v
    #the core part of the code is the same with the exception that as the output of the while 
    #I have reached the destination node
    while current_node != end:
        adjacentes = adjacent[current_node]
        weight_to_current_node = visited[current_node][1]
        for node in adjacentes:
            weight = graph[current_node][node][p] + weight_to_current_node
            if node not in visited:
                visited[node] = (current_node, weight)
                hp.heappush(F,(weight,node))
            else:
                current_shortest_weight = visited[node][1]
                if current_shortest_weight > weight:
                    visited[node] = (current_node, weight)
                    hp.heappush(F,(weight,node))
        #also here I check if the border is empty, if it's empty and we haven't left yet while it means that 
        #I checked all the nodes without having reached the destination node 
        #so I can say that there is no path between the starting node and the one of destination
        if not F:
            return "Route Not Possible"
        current_node = hp.heappop(F)[1]
    # Work backwards between the destinations visited up to the node that has no parent set to None, then up to the
    #first node. I start from the current node because after the while the current node is the destination node
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = visited[current_node][0]
        current_node = next_node
     # having added the nodes in the list from the last one obviously before returning it I reverse it
    path.reverse()
    return path

def order_walk(v,nodes,p):
    path = dijkstra_modified(v,nodes[0],p)
    #here I check if the output of my function is a string if it is a string 
    #it means that there is no path between the two nodes and I simply return it.
    if type(path) == str:
        return path
    for i in range(1,len(nodes)):
        #concatenate the paths between the pairs of nodes to create the ordered walk
        path1 = dijkstra_modified(nodes[i-1],nodes[i],p)
        #I repeat the check if there is a path or not
        if type(path1) == str:
            return path1
        else:
            path += path1[1:]
    #return final walk
    return path

## Funtion 4

def dijkstra_modified2(v,end,p,graph = G,adjacent = adj):
    # v is the start node
    # end is the destination node
    # Also here as in the functionality 1 I use djikstra with small changes here instead of calculating 
    # the neighbors of the starting node I calculate the minimum path between the starting node and the arrival node.
    # I create a dictionary of the visited ones but besides the weight I save the predecessor
    # that then it will be useful to me to reconstruct the path
    visited = {v: (None, 0)}
    F = []
    hp.heapify(F)
    current_node = v
    #the core part of the code is the same with the exception that as the output of the while 
    #I have reached the destination node
    while current_node != end:
        adjacentes = adjacent[current_node]
        weight_to_current_node = visited[current_node][1]
        for node in adjacentes:
            weight = graph[current_node][node][p] + weight_to_current_node
            if node not in visited:
                visited[node] = (current_node, weight)
                hp.heappush(F,(weight,node))
            else:
                current_shortest_weight = visited[node][1]
                if current_shortest_weight > weight:
                    visited[node] = (current_node, weight)
                    hp.heappush(F,(weight,node))
        #also here I check if the border is empty, if it's empty and we haven't left yet while it means that 
        #I checked all the nodes without having reached the destination node 
        #so I can say that there is no path between the starting node and the one of destination
        if not F:
            return "Route Not Possible"
        current_node = hp.heappop(F)[1]
    # Work backwards between the destinations visited up to the node that has no parent set to None, then up to the
    #first node. I start from the current node because after the while the current node is the destination node
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = visited[current_node][0]
        current_node = next_node
     # having added the nodes in the list from the last one obviously before returning it I reverse it
    path.reverse()
    return path

#this exercise is a typical theoretical infromatics problem known as TSP, the exact calculation
#of this problem is computationally difficult, therefore we tend to solve it using heuristics that 
#return a solution close to the optimal, we have decided to implement the heuristics known as nearest neighbor, 
#which finds the shortest path between the nearest nodes until all nodes are visited
import math
def func4(H,nodes,p):
    mini = math.inf
    path1 = []
    # iterate until all the nodes are visited
    while nodes:
        for node in nodes:
            # calculate all the distances from my starting point and all the elements to visit
            path,pe = dijkstra_modified2(H,node,p,graph = G,adjacent = adj)
            peso = pe[node][1]
            if peso < mini:
                mini = peso
                # i get closer node and its relative path
                X = (node,path)
        mini = math.inf
        # I add the path relative to the final path
        if len(path1) == 0:
            #if path is a string as it was implemented djikstra return the string that tells me there is no path
            if type(X[1]) == str:
                return X[1]
            else:
                path1 += X[1]
        else:
            if type(X[1]) == str:
                return X[1]
            else:
                path1 += X[1][1:]
        # I update my starting node which was my arrival in the previous path and 
        #I remove it from the set of nodes to visit
        H = X[0]
        nodes.remove(X[0])
    return path1
    

#Visualization


def mymap(nodelst):      
    pos = G.nodes[nodelst[0]] # for the start point
    
    # creating the Map focusing on the start point

    vismap = folium.Map(location=[pos['latitude'], pos['longitude']], zoom_start=10)
    
    #Layers Map
    
    folium.raster_layers.TileLayer('Open Street Map').add_to(vismap)
    folium.raster_layers.TileLayer('Stamen Terrain').add_to(vismap)
    
    #adding lcontrol on map
    folium.LayerControl().add_to(vismap)
    
    #minimap
    
    
    # plugin for mini map
    visminimap = plugins.MiniMap(toggle_display=True)

    # add minimap to map
    vismap.add_child(visminimap)

    # add scroll zoom toggler to map
    plugins.ScrollZoomToggler().add_to(vismap)

    # creating a marker of HOme on the start point
    
    folium.Marker(location=[(pos['latitude']),(pos['longitude'])],
                  icon=folium.Icon(color='red', icon='home'), popup = (nodelst[0])).add_to(vismap)

    # creating a marker on the rest of the point
    
    for i in range (len(nodelst)-1):
        pos = (G.nodes[nodelst[i+1]])
        folium.Marker(location=[(pos['latitude']),(pos['longitude'])],popup = (nodelst[i+1])).add_to(vismap)
    
    return vismap

# a is the list of nodes
# map_name is the map with nodes already generated

def map_routes(lst,map_name):
# making list of all coordinates of the NODES avaialble in a
    for t in range (len(lst)):
        cordlst = 0
        cordlst = []
        a = (lst[t])             
        for i in a:
            cordlst.append(list(G.nodes[i].values())) 
        plugins.AntPath(cordlst).add_to(map_name)
    return map_name

def map_routesf3(lst,map_name):
# making list of all coordinates of the NODES avaialble in a
    cordlst = 0
    cordlst = []            
    for i in lst:
        cordlst.append(list(G.nodes[i].values())) 
    plugins.AntPath(cordlst).add_to(map_name)
    return map_name


# a is the list of nodes
# map_name is the map with nodes already generated
       
def map_routespoly(lst,map_name):
# making list of all coordinates of the NODES avaialble in a
    for t in range (len(lst)):
        cordlst = 0
        cordlst = []
        a = (lst[t])             
        for i in a:
            cordlst.append(list(G.nodes[i].values())) 
        folium.vector_layers.PolyLine(cordlst, color = 'red').add_to(map_name)
    return map_name

def circlemarker(lst,map_name): # Not neighbor
# making list of all coordinates of the NODES avaialble in a
    for i in range (len(lst)):
        pos = (G.nodes[lst[i]])
        folium.CircleMarker(location=[(pos['latitude']),(pos['longitude'])], radius=10, color='blue', fill_color='red',
                  popup = (lst[i])).add_to(map_name)
    return map_name


def circlemarkerf3(lst,map_name): 
# making list of all coordinates of the NODES avaialble in a
    for i in range (len(lst)):
        pos = (G.nodes[lst[i]])
        folium.Marker(location=[(pos['latitude']),(pos['longitude'])],icon=folium.Icon(color='red'),
                  popup = (lst[i])).add_to(map_name)
    return map_name


#Front END
####################################################################################
from tkinter import *
from tkinter import messagebox
root = Tk()

# Background

background_image = PhotoImage(file = 'HW5IMAGE.png')
background_label = Label(root, image=background_image)

# Gridding Background
background_label.grid(rowspan = 25, column = 9, sticky = N)

###################################################################
## Labels
# LabelF1
labelf1 = Label(root, text = 'Functionality 1 - Find the Neighbours!', font = 'bold')
labelf1a = Label(root, text = 'Enter the Home Node ')
labelf1b = Label(root, text = 'Enter the Parametere time / distance ')
labelf1c = Label(root, text = 'Enter the thrushold for above parameter ')

# LabelF2
labelf2 = Label(root, text = 'Functionality 2 - Find the smartest Network!', font = 'bold')
labelf2a = Label(root, text = 'Enter the visiting Node seperated by (,)')
labelf2b = Label(root, text = 'Enter the Parametere time / distance ')

# LabelF3 
labelf3 = Label(root, text = 'Functionality 3 - Shortest Ordered Route', font = 'bold')
labelf3a = Label(root, text = 'Enter the Home Node ')
labelf3b = Label(root, text = 'Enter the visiting Node seperated by (,)')
labelf3c = Label(root, text = 'Enter the Parametere time / distance ')


# LabelF4
labelf4 = Label(root, text = 'Functionality 4 - Shortest Route', font = 'bold')
labelf4a = Label(root, text = 'Enter the Home Node ')
labelf4b = Label(root, text = 'Enter the visiting Node seperated by (,) ')
labelf4c = Label(root, text = 'Enter the Parametere time / distance ')

#empty Label

empty_label1 =  Label(root, text = '#')
empty_label2 =  Label(root, text = '#')
empty_label3 =  Label(root, text = '#')
empty_label4 =  Label(root, text = '#')
###################################################################
## Entry
#EntryF1
Entryf1a = Entry(root)
Entryf1b = Entry(root)
Entryf1c = Entry(root)

#EntryF2
Entryf2a = Entry(root)
Entryf2b = Entry(root)

#EntryF3
Entryf3a = Entry(root)
Entryf3b = Entry(root)
Entryf3c = Entry(root)

#EntryF4
Entryf4a = Entry(root)
Entryf4b = Entry(root)
Entryf4c = Entry(root)

#####################################################################3
## Buttons

buttonF1 = Button(root, text="Generate Function 1", fg="red")
buttonF2 = Button(root, text="Generate Function 2", fg="red")
buttonF3 = Button(root, text="Generate Function 3", fg="red")
buttonF4 = Button(root, text="Generate Function 4", fg="red")

################################################################33
## Griding
# GridingF1
labelf1.grid(row = 0, sticky = W)
labelf1a.grid(row =1, sticky = E)
labelf1b.grid(row =2, sticky = E)
labelf1c.grid(row =3, sticky = E)

Entryf1a.grid(row=1, column = 1)
Entryf1b.grid(row=2, column = 1)
Entryf1c.grid(row=3, column = 1)

empty_label1.grid(row=4, sticky = W)

buttonF1.grid(row = 0, column = 3, columnspan = 4)

# GridingF2
labelf2.grid(row = 6, sticky = W)
labelf2a.grid(row = 7, sticky = E)
labelf2b.grid(row =8, sticky = E)

Entryf2a.grid(row=7, column = 1)
Entryf2b.grid(row=8, column = 1)

empty_label2.grid(row=9, sticky = W)

buttonF2.grid(row = 6, column = 3, columnspan = 4)

# GridingF3

labelf3.grid(row = 11, sticky = W)
labelf3a.grid(row = 12, sticky = E)
labelf3b.grid(row = 13, sticky = E)
labelf3c.grid(row = 14, sticky = E)

Entryf3a.grid(row=12, column = 1)
Entryf3b.grid(row=13, column = 1)
Entryf3c.grid(row=14, column = 1)


empty_label3.grid(row=15, sticky = W)

buttonF3.grid(row = 11, column = 3, columnspan = 4)

# GridingF4

labelf4.grid(row = 17, sticky = W)
labelf4a.grid(row = 18, sticky = E)
labelf4b.grid(row = 19, sticky = E)
labelf4c.grid(row = 20, sticky = E)

Entryf4a.grid(row=18, column = 1)
Entryf4b.grid(row=19, column = 1)
Entryf4c.grid(row=20, column = 1)


empty_label4.grid(row=21, sticky = W)

buttonF4.grid(row = 17, column = 3, columnspan = 4)

# Button Clicks F1
def f_1(event):
    nodelstf1 = 0
    v = int(Entryf1a.get())
    w = str(Entryf1b.get())
    d = int(Entryf1c.get())

    nodelstf1 = neighbors(v,w,d,graph = G, adjacent = adj)
    omap = mymap(nodelstf1[0])

    # Ant PATH ROUTE FOR NEIGHBOURS (CONDITION)
    map_routes(nodelstf1[1],omap)

    # POLY PATH ROUTES FOR NOT NEIGHBOURS(CONDITION)
    map_routespoly(nodelstf1[2],omap)

    # CIRCLE MARKER FOR NOT NEIGHBORS
    circlemarker(nodelstf1[3],omap)

    return omap.save('F1map.html')

buttonF1.bind("<Button-1>", f_1)

# Button Clicks F2
def f_2(event):
    f2k = (Entryf2a.get()).split(",")
    f2p = str(Entryf2b.get())
    f2k = list(map(int, f2k))
    f2k = set(f2k)
    return fun2_visual(f2k, f2p).save('F2map.html')

buttonF2.bind("<Button-1>", f_2)


# Button Clicks F3
def f_3(event):
    f3s = int(Entryf3a.get())
    f3k = (Entryf3b.get()).split(",")
    f3p = str((Entryf3c.get()))
    f3k = [int(i) for i in f3k]
    nodelstf3 = order_walk(f3s,f3k,f3p)
    refpoint = set(nodelstf3)-set(f3k)
    refpoint = list(refpoint)
    cm = mymap(refpoint)
    dm = circlemarkerf3(f3k,cm)
    return map_routesf3(nodelstf3,dm).save('F3map.html')

buttonF3.bind("<Button-1>", f_3)


# Button Clicks F4
def f_4(event):
    f4s = int(Entryf4a.get())
    f4k = (Entryf4b.get()).split(",")
    f4p = str((Entryf4c.get()))
    f4k = list(map(int, f4k))
    f4k = set(f4k)
    nodelstf4 = func4(f4s,f4k,f4p)
    cm = mymap(nodelstf4)
    return map_routesf3(nodelstf4,cm).save('F4map.html')


buttonF4.bind("<Button-1>", f_4)


root.mainloop()