import sys
import numpy as np
import automaton


class Vertex:
    """The Vertex class generates the nodes of the search  graph as objects 
    """
    def __init__(self, node):
        # We use the coordinates of a node/cell in form of a tuple "(y,x)"" as the id of a vertex object
        self.id = node
        # Stores the neighbors of a cell
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxsize
        # Mark all nodes unvisited        
        self.visited = False  
        # Stores Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        """Adds neighbour to the Vertex

        :param neighbor: Neighbour node of Vertex
        :type neighbor: Vertex
        :param weight: Edge weight to the node, defaults to 0
        :type weight: float
        """
        self.adjacent[neighbor] = weight

    def get_connections(self):
        """Returns all Neighbours of the Vertex from the dictionary adjecent

        :return: Neighbour Vertices of the object
        :rtype: Vertex
        """
        return self.adjacent.keys()  

    def get_id(self):
        """
        :return: id of Vetrex object
        :rtype: tuple, (y, x)
        """
        return self.id

    def get_weight(self, neighbor):
        """Gives weight of neighbour object

        :param neighbor: Neighbour of Vertex
        :type neighbor: Vertex
        :return: weight of edge to neighbour
        :rtype: float
        """
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        """
        :param dist: Distance from initial node to reach thid Vertex
        :type dist: float
        """
        self.distance = dist

    def get_distance(self):
        """
        :return: Distance from initial node to reach thid Vertex
        :rtype: float
        """
        return self.distance

    def set_previous(self, prev):
        """ Set the previous node which comes from the shortest distance regarding to the initial node

        :param prev: Previous node
        :type prev: Vertex
        """
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        """Returns ID and all IDs of the neighbour nodes
        """
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    """Can create objects of class Graph
    """
    def __init__(self):
        """Attributes store all Vertices and the total amount of Vertecies
        """
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        """Iterates through the Vertices of Graph an returns one random Vertex
        """
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        """Creates a Vertex object with id node and adds it to the dictionary of the Graph's Vertices
        """
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        """Gives back the object Vertex n or None if Vertex n is not in the Graph
        """
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        """Adds connection from one Vertex to another

        :param frm: form Vertex
        :type frm: Vertex
        :param to: to Vertex
        :type to: Vertex
        :param cost: distance between nodes, defaults to 0
        :type cost: float
        """
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        """Returns Vertecies of Graph
        """
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

#For queue of the search process
import heapq

def dijkstra(aGraph, start, target):
    """[summary]

    :param aGraph: Search Graph
    :type aGraph: Graph
    :param start: Start node
    :type start: Vertex
    :param target: target
    :type target: Vertex
    """
    #print("Dijkstra's shortest path")
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(),v.get_id(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[2]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v.get_id(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)
        #reset distances 
        

def generate_graph(states):
    """Maps cell field to Graph

    :param states: matrix of cell field P=1, O=2, T=3, E=0,
    :type states: matrix int 
    :return: Graph with correct set previous nodes
    :rtype: Graph
    """
    g = Graph()
    y_shape = states.shape[0]
    x_shape = states.shape[1]
    offset = [[1,0],[0,1],[-1,0],[0,-1]]
    #add vertecies 
    for x in range(x_shape):
        for y in range(y_shape):
            if states[y,x] == automaton.CellState.O.value:
                continue
            g.add_vertex((y, x))

    #add edges
    all_nodes = g.vert_dict.keys()
    for node in all_nodes:
        neighbors = np.add(node,offset)
        for neighbor in neighbors:
            if tuple(neighbor) in all_nodes:
                g.add_edge(node,tuple(neighbor),0.5)
    return g
