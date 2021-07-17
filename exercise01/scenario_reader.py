import json
import automaton
from automaton import Parameters
from automaton import Scenario


def read_file(filename):
    """returns a deserialized json object representing the specification of a scenario

    :param filename: path to json file
    :type filename: string
    :raises RuntimeError: error if file not found or not deserializable
    :return: json object
    :rtype: json
    """
    try:
        f = open(filename,)
        scenario_data = json.load(f)
        f.close()
        return scenario_data
    except:
        raise RuntimeError('JSON Scenario file could not be read in. Please check the file for correctness and that it exists.')

def make_scenario(filename):
    """returns a scenario specified by a given json file

    :param filename: path to file to be read in
    :type filename: string
    :return: complete scenario
    :rtype: Scenario
    """
    #read json file
    scenario_data = read_file(filename)
    #create paramters object
    parameters = Parameters(scenario_data['width'], scenario_data['height'], scenario_data['cell_width'],scenario_data['r_max'], scenario_data['speed_pedestrian'], scenario_data['speed_control'], scenario_data['use_dijkstra'])
    #instantiate new scenario with parameter object
    scenario = Scenario(parameters)
    #get data from JSON File
    pedestrians_coordinates_array = scenario_data["pedestrians_coordinates"]
    pedestrians_for_density_array = scenario_data["pedestrians_for_density"]
    obstacles_coordinates_array = scenario_data["obstacles_coordinates"]
    obstacle_rectangular_array = scenario_data["obstacle_rectangular"]
    target_coordinates = scenario_data["target_coordinates"]
    #adding single pedestrians
    for pedestrian in pedestrians_coordinates_array:
        scenario.add_pedestrian(pedestrian["x"], pedestrian["y"])
    #adding pedestrians for density
    for pedestrians in pedestrians_for_density_array:
        scenario.add_pedestrians_for_density(pedestrians["density"], pedestrians["upper_left_x"], pedestrians["upper_left_y"], pedestrians["lower_right_x"], pedestrians["lower_right_y"])
    #adding single obstacles
    for obstacle in obstacles_coordinates_array:
        scenario.add_obstacle(obstacle["x"], obstacle["y"])
    #adding rectangular obstacles
    for obstacles in obstacle_rectangular_array:
        scenario.add_obstacle_rectangular(obstacles["upper_left_x"], obstacles["upper_left_y"], obstacles["lower_right_x"], obstacles["lower_right_y"])
    #setting target
    scenario.set_target(target_coordinates["x"], target_coordinates["y"])
    return scenario
