import numpy as np
from enum import Enum
import random
import math
import dijkstra
from dijkstra import dijkstra
from dijkstra import generate_graph
from dijkstra import shortest

class CellState(Enum):
    E = 0
    P = 1
    O = 2
    T = 3


class Parameters:
    """this class acts as a place for settings for the simulation
    """
    def __init__(self, width=4, height=3, cell_width=0.5, r_max=0.6, speed_pedestrian=1.0, speed_control=True, use_dijkstra=False):
        self.width = width
        self.height = height
        self.cell_width = cell_width #in m
        self.speed_pedestrian = speed_pedestrian #in m/s
        self.speed_control = speed_control #true: all pedestrians walk with same speed in arbitrary direction
        self.utility_target = 1000
        self.utility_obstacle = 1000
        self.utility_pedestrian = 1000
        self.utility_pedestrian_reach = r_max #r_max from th exercise sheet
        self.use_dijkstra = use_dijkstra #if to use dijkstra for distance calculation or euclidean distance

class Scenario:
    """this class is used to set up the the initial state of the simulation
        including adding of pedestrians, obstacles and a target

        Note: the indexing in this class is user-friendly and thus x,y- based and 1-based
    """
    def __init__(self, parameters=Parameters()):
        num_cells_x = math.floor(parameters.width / parameters.cell_width)
        num_cells_y = math.floor(parameters.height / parameters.cell_width)
        self.states = np.zeros((num_cells_y, num_cells_x))
        self.distances = np.zeros((num_cells_y, num_cells_x))
        self.utilities = np.zeros((num_cells_y, num_cells_x))
        self.target = None
        self.pedestrian_positions = []
        self.parameters = parameters
    
    def add_pedestrian(self, x, y):
        """adds a pedestrian to the scenario

        :param x: x coordinate of the pedestrian
        :type x: int
        :param y: y coordinate of the pedestrian
        :type y: int
        :raises ValueError: Error if coordinates not on grid.
        """
        if not self.is_cell_on_grid(y - 1, x - 1):
            raise ValueError('Failed to add Pedestrian. Cell not on Grid.')
        else:
            #append pedestrian only if it doesn't exist yet
            if not [y - 1, x - 1] in self.pedestrian_positions:
                self.states[y - 1, x - 1] = CellState.P.value
                self.pedestrian_positions.append([y - 1, x - 1])
                self.calculate_utilities_for_pedestrian(y - 1, x - 1)
                return True
            return False


    def add_pedestrians_for_density(self, density, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
        """adds pedestrians to a given reactangular with a given density in pedestrians / m^2

        :param density: the desired density in pedestrians / m^2
        :type density: float
        :param upper_left_x: upper left x coordinate of recangle
        :type upper_left_x: int
        :param upper_left_y: upper left y coordinate of rectangle
        :type upper_left_y: int
        :param lower_right_x: lower right coordinate of rectangle
        :type lower_right_x: int
        :param lower_right_y: lower right coordinate of rectangle
        :type lower_right_y: int
        :raises ValueError: error if cell not on grid
        :raises ValueError: error if density to high
        """
        if not (self.is_cell_on_grid(upper_left_y - 1, upper_left_x - 1) and self.is_cell_on_grid(lower_right_y - 1, lower_right_x - 1)):
            raise ValueError('Failed to add Pedestrians. Cell not on Grid.')
        else:
            num_cells = (lower_right_x - upper_left_x + 1) * (lower_right_y - upper_left_y + 1)
            num_pedestrians_to_add = math.floor((num_cells * (self.parameters.cell_width ** 2)) * density)
            if num_pedestrians_to_add > num_cells:
                raise ValueError('Failed to add Pedestrians. Density to high.')
            num_added_pedestrians = 0
            while num_added_pedestrians != num_pedestrians_to_add:
                rand_y = random.randrange(upper_left_y, lower_right_y + 1)
                rand_x = random.randrange(upper_left_x, lower_right_x + 1)
                if self.add_pedestrian(rand_x, rand_y):
                    num_added_pedestrians += 1      

    def calculate_euclidean(self, i, j, k, l):
        """euclidean distance measure between two cells on the grid

        :param i: y coordinate of first cell
        :type i: int
        :param j: x coordinate of first cell
        :type j: int
        :param k: y coordinate of second cell
        :type k: int
        :param l: x coordinate of second cell
        :type l: int
        :return: euclidean distance
        :rtype: float
        """
        return math.sqrt((i - k) ** 2 + (j - l) ** 2) * self.parameters.cell_width

    def calculate_cost_function(self, radius, radius_max):
        """actual cost function according to exercise sheet but multiplied with factor of 100
        """
        if radius < radius_max:
            return np.exp(1 / (radius ** 2 - radius_max ** 2)) * 100
        else:
            return 0

    def get_neighbor_for_radius(self, start_i, start_j, radius):
        """helper function to calculate all valid neighbors of a cell within a given radius in meters

        :param start_i: i coordinate of cell
        :type start_i: int
        :param start_j: j coordinate of cell
        :type start_j: int
        :param radius: radius around cell in meter
        :type radius: float
        :return: list of coordinates of neighbors within radius
        :rtype: list
        """
        neighbor_within_radius = []
        #calculate neighbors (also outside of grid) in first quadrant (negative i direction, positive j direction)
        first_quadrant_neighbors = []
        for j in range(math.floor(radius / self.parameters.cell_width) + 1):
            right_neighbor = np.add([start_i, start_j], [0, j])
            for i in range(math.floor(radius / self.parameters.cell_width) + 1):
                upper_neighbor = np.add(right_neighbor, [-i,0])
                if (upper_neighbor[0] != start_i or upper_neighbor[1] != start_j) and (self.calculate_euclidean(upper_neighbor[0], upper_neighbor[1], start_i, start_j) <= radius):
                    first_quadrant_neighbors.append([upper_neighbor[0], upper_neighbor[1]])
        #mirror found neighbors to other quadrants and add those cells also if they are on the grid
        for neighbor in first_quadrant_neighbors:
            neighbor_second = np.add([start_i, start_j], [-(neighbor[0] - start_i), neighbor[1] - start_j])
            neighbor_third = np.add([start_i, start_j], [-(neighbor[0] - start_i), -(neighbor[1] - start_j)])
            neighbor_fourth = np.add([start_i, start_j], [(neighbor[0] - start_i), -(neighbor[1] - start_j)])
            if self.is_cell_on_grid(neighbor[0], neighbor[1]):
                neighbor_within_radius.append([neighbor[0], neighbor[1]])
            if self.is_cell_on_grid(neighbor_second[0], neighbor_second[1]) and [neighbor_second[0], neighbor_second[1]] not in neighbor_within_radius:
                neighbor_within_radius.append([neighbor_second[0], neighbor_second[1]])
            if self.is_cell_on_grid(neighbor_third[0], neighbor_third[1]) and [neighbor_third[0], neighbor_third[1]] not in neighbor_within_radius:
                neighbor_within_radius.append([neighbor_third[0], neighbor_third[1]])
            if self.is_cell_on_grid(neighbor_fourth[0], neighbor_fourth[1]) and [neighbor_fourth[0], neighbor_fourth[1]] not in neighbor_within_radius:
                neighbor_within_radius.append([neighbor_fourth[0], neighbor_fourth[1]])
        return neighbor_within_radius

    def calculate_utilities_for_pedestrian(self, i, j):
        """adds cost value to utilities for surrounding neighbor cells according to cost function
        """
        neighbors = self.get_neighbor_for_radius(i, j, self.parameters.utility_pedestrian_reach)
        #add maximum cost on cell where pedestrian stands
        self.utilities[i,j] += self.parameters.utility_pedestrian
        #add cost of costfunction for neighboring cells
        for neighbor in neighbors:
            radius = self.calculate_euclidean(neighbor[0], neighbor[1], i, j)
            self.utilities[neighbor[0], neighbor[1]] += self.calculate_cost_function(radius, self.parameters.utility_pedestrian_reach)
    
    def remove_utilities_for_pedestrian(self, i, j):
        """removes cost value to utilities for surrounding neighbor cells according to cost function
        """
        neighbors = self.get_neighbor_for_radius(i, j, self.parameters.utility_pedestrian_reach)
        #remove maximum cost on cell where pedestrian stands
        self.utilities[i,j] -= self.parameters.utility_pedestrian
        #remove cost of costfunction for neighboring cells
        for neighbor in neighbors:
            radius = self.calculate_euclidean(neighbor[0], neighbor[1], i, j)
            self.utilities[neighbor[0], neighbor[1]] -= self.calculate_cost_function(radius, self.parameters.utility_pedestrian_reach)

    def add_obstacle(self, x, y):
        """adds an obstacle to the scenario

        :param x: x coordinate of the obstacle
        :type x: int
        :param y: y coordinate of the obstacle
        :type y: int
        :raises ValueError: Error if coordinates not on grid.
        """
        if not self.is_cell_on_grid(y - 1, x - 1):
            raise ValueError('Failed to add Obstacle. Cell not on Grid.')
        else:
            self.states[y - 1, x - 1] = CellState.O.value
            self.utilities[y - 1, x - 1] += self.parameters.utility_obstacle

    def add_obstacle_rectangular(self, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
        """adds an obstacle in reactangular shape to the scenario

        :param upper_left_x: upper left x coordinate of recangle
        :type upper_left_x: int
        :param upper_left_y: upper left y coordinate of rectangle
        :type upper_left_y: int
        :param lower_right_x: lower right coordinate of rectangle
        :type lower_right_x: int
        :param lower_right_y: lower right coordinate of rectangle
        :type lower_right_y: int
        :raises ValueError: Error if coordinates not on grid.
        """
        if not (self.is_cell_on_grid(upper_left_y - 1, upper_left_x - 1) and self.is_cell_on_grid(lower_right_y - 1, lower_right_x - 1)):
            raise ValueError('Failed to add Rectangular Obstacle. Cell not on Grid.')
        else:
            for x_offset in range(lower_right_x - upper_left_x + 1):
                for y_offset in range(lower_right_y - upper_left_y + 1):
                    self.add_obstacle(upper_left_x + x_offset, upper_left_y + y_offset)

    def set_target(self, x, y):
        """adds a target to the scenario. If a target was already set, the old one will be deleted

        :param x: x coordinate of the target
        :type x: int
        :param y: y coordinate of the target
        :type y: int
        :raises ValueError: Error if coordinates not on grid.
        """
        if not self.is_cell_on_grid(y - 1, x - 1):
            raise ValueError('Failed to add Target. Cell not on Grid.')
        else:
            if self.target is not None:
                self.states[self.target[0] - 1, self.target[1] - 1] = CellState.E.value
            self.target = np.array([y - 1, x - 1])
            self.states[y - 1, x - 1] = CellState.T.value
            self.utilities[y - 1, x - 1] += self.parameters.utility_target

    def is_cell_on_grid(self, i, j):
        """checks if a cell specified by its row and column index is on the grid
        """
        return -1 < i < self.states.shape[0] and -1 < j < self.states.shape[1]

    def print_distances(self):
        for i in range(self.distances.shape[0]):
            for j in range(self.distances.shape[1]):
                print(round(self.distances[i, j], 1)," ", end =" ")
            print("\n")
    
    def print_utilities(self):
        for i in range(self.utilities.shape[0]):
            for j in range(self.utilities.shape[1]):
                print(round(self.utilities[i, j], 1)," ", end =" ")
            print("\n")

class Automaton:
    """this class is instantiated with a scenario and represents the cellular automaton
    it is used to simulate the movement of pedestrians

        Note: the indexing in this class is programmer-friendly and thus i,j- based and 0-based
    """
    def __init__(self, scenario, measure_speed=False, update_pedestrians_shuffled=False):
        """initializes the cellular automaton with a given scenario

        :param scenario: a predefined scenario
        :type scenario: Scenario
        :raises Exception: Error if no target exists in Scenario
        """
        self.scenario = scenario
        self.measure_speed = measure_speed
        self.update_pedestrians_shuffled = update_pedestrians_shuffled
        if scenario.target is None:
            raise Exception('Failed to instantiate Automaton. No target available in scenario.')
        else:
            if self.scenario.parameters.use_dijkstra:
                self.calculate_distances_dijkstra()
            else:
                self.calculate_distances_euclidean()

    def calculate_distances_euclidean(self):
        """fills the 2d distance array of the scenario with distance values according to the euclidean distance
        """
        k, l = self.scenario.target[0], self.scenario.target[1]
        for i in range(self.scenario.distances.shape[0]):
            for j in range(self.scenario.distances.shape[1]):
                #compute euclidean distance of each cell to target cell
                if self.scenario.states[i, j] != CellState.O.value:
                    self.scenario.distances[i, j] += math.sqrt((i - k) ** 2 + (j - l) ** 2) * self.scenario.parameters.cell_width
                    self.scenario.utilities[i, j] += math.sqrt((i - k) ** 2 + (j - l) ** 2) * self.scenario.parameters.cell_width

    def calculate_distances_dijkstra(self):
        """fills the 2d distance array of the scenario with distance values according to the dijkstra path search
        """
        states = self.scenario.states
        target = tuple(self.scenario.target) 
        for x in range(states.shape[1]):
            for y in range(states.shape[0]):
                if (states[y,x] == 2) or (states[y,x] == 3) or (self.scenario.distances[y,x] != 0):
                    continue 
                graph = generate_graph(states)
                dijkstra(graph, graph.get_vertex((y,x)), graph.get_vertex(target))
                path = [target]
                shortest(graph.get_vertex(target), path)
                path = np.array(path)
                path_length = (path.shape[0] - 1) * self.scenario.parameters.cell_width
                for node in path[::-1]:
                    node = np.array(node)
                    if self.scenario.distances[node[0], node[1]] == 0:
                        self.scenario.distances[node[0],node[1]] += path_length
                        self.scenario.utilities[node[0],node[1]] += path_length
                    path_length -= self.scenario.parameters.cell_width

    def move_pedestrians(self):
        """function that represents one time step of the simulation within each pedestrian can walk one cell as maximum

        :raises RuntimeError: error if no pedestrian moved at all
        """
        states_changed = False
        
        neighborOffsets = [ [-1,0], [0,1], [1,0], [0,-1] ] #von Neumann neighbborhood
        neighborOffsetsWithDiagonals = [ [-1,0], [0,1], [1,0], [0,-1], [1, 1], [-1, -1], [-1, 1], [1, -1] ] #moore neighborhood

        #better simulation if update pedestrians not always in the same order
        if self.update_pedestrians_shuffled:
            random.shuffle(self.scenario.pedestrian_positions)

        #iterate over pedestrians
        for i, pedestrian in enumerate(self.scenario.pedestrian_positions):
            #remove own utility values to not influence own penalties when choosing next neighbor cell
            self.scenario.remove_utilities_for_pedestrian(pedestrian[0], pedestrian[1]) 
            #set initial minimum utitlity to distance value of current cell
            minUtility = self.scenario.utilities[pedestrian[0], pedestrian[1]]
            #set initial next neighbor to current cell
            nextNeighbor = pedestrian
            #if speed control is activated, let pedestrians also sometimes move diagonally
            if self.scenario.parameters.speed_control is True:
                #use factor of around 50*sqrt(2) to compensate longer steplength when walking diagonally
                offsets = neighborOffsets if random.randrange(100) < 70 else neighborOffsetsWithDiagonals
            else:
                offsets = neighborOffsets
            #iterate over neighbors
            for offset in offsets:
                #create neighbor
                neighbor = np.add([pedestrian[0], pedestrian[1]], offset)
                #update next neighbor if neighbor in grid and its utility lower than current minUtility
                if self.scenario.is_cell_on_grid(neighbor[0], neighbor[1]) and self.scenario.utilities[neighbor[0], neighbor[1]] < minUtility:
                    minUtility = self.scenario.utilities[neighbor[0], neighbor[1]]
                    nextNeighbor = neighbor

            if not (nextNeighbor[0] == pedestrian[0] and nextNeighbor[1] == pedestrian[1]):
                states_changed = True
                #go to nextneighbor
                self.scenario.states[pedestrian[0], pedestrian[1]] = CellState.E.value
                self.scenario.states[nextNeighbor[0], nextNeighbor[1]] = CellState.P.value

                if self.measure_speed:
                    #if pedestrian is moving, check if the next cell is in the measuring points range
                    if(nextNeighbor[1] >= self.measuring_points_boundaries[0] and nextNeighbor[1] <= self.measuring_points_boundaries[1]):
                        self.pedestrians_speed_sum[i] += self.moving_speed #increase speed
                        self.pedestrians_time_in_measuring_points[i] += 1 #increase timestamps in measuring points area

                #update pedestrian position in pedestrian list
                self.scenario.pedestrian_positions[i] = [nextNeighbor[0], nextNeighbor[1]]
                #update also utilities

                self.scenario.calculate_utilities_for_pedestrian(nextNeighbor[0], nextNeighbor[1])
            else:
                self.scenario.calculate_utilities_for_pedestrian(pedestrian[0], pedestrian[1])

                if self.measure_speed:
                    #check if pedestrians is standing within the measuring area
                    if(nextNeighbor[1] >= self.measuring_points_boundaries[0] and nextNeighbor[1] <= self.measuring_points_boundaries[1]):
                        self.pedestrians_time_in_measuring_points[i] += 1 #increase timestamps in measuring points area, speed sum stayes same


        if states_changed == False:
            raise RuntimeError('Cannot move pedestrians since there are no pedestrians in the scenario or all reached their target!')


    def simulate_pedestrians(self, time_steps = None):
        """actual simulation of cellular automaton. calles move_pedestrians at most time_steps times

        :param time_steps: the maximum number of time steps
        :type time_steps: int
        :return: list of scenario states representing the different time steps of the automaton
        :rtype: list
        """

        #initialize empty list
        data = []
        #set time_steps
        if time_steps is None:
            #maximum number of possible steps is the number of cells on the grid
            time_steps = self.scenario.states.shape[0] * self.scenario.states.shape[1]
        #loop over time_steps
        for k in range(time_steps):
            #add current scenario states array to final data list
            data.append(np.copy(self.scenario.states))
            try:
                #manipulate scenario states
                self.move_pedestrians()
            except RuntimeError:
                #stop simulation if no pedestrian was moved at all
                print("Simulation stopped after", k + 1 ,"simulation steps, since all pedestrians are already at their target.")
                break
        print("Finished simulating", k + 1 ,"simulation steps.")
        #if speed measure is done, calculate mean speed
        if self.measure_speed:
            self.pedestrians_average_speed = self.pedestrians_speed_sum / self.pedestrians_time_in_measuring_points
            self.average_speed = np.sum(self.pedestrians_average_speed) / len(self.scenario.pedestrian_positions)
        
        return data


    def set_measuring_point(self, x_left, x_right, density, moving_speed):
        """sets a speed measurement area over the whole height of the scenario and from x_left to x_right

        :param x_left: x coordinate of begin of area
        :type x_left: int
        :param x_right: x coordinate of end of area
        :type x_right: int
        :param density: the density for which the velocity to be measured
        :type density: float
        :param moving_speed: the free walking speed of pedestrians
        :type moving_speed: float
        """
        self.measuring_points_boundaries = [x_left, x_right]
        self.density = density
        self.pedestrians_speed_sum = np.zeros(len(self.scenario.pedestrian_positions))
        self.pedestrians_time_in_measuring_points = np.zeros(len(self.scenario.pedestrian_positions))
        self.moving_speed = moving_speed
        self.measure = 0
        self.measure_all = 0        
