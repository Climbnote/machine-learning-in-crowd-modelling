import json
import numpy as np


def get_uniform_pedestrian_coords_for_corridor(size):
    """get the coordinates of size pedestrians for the corridor scenario 

    :param size: number of pedestrians
    :type size: int
    :return: tupel of x and y coordinates within the bounding box for pedestrians in the bootleneck scenario
    """
    #bounding box of pedestrians
    xlim = (7.1, 12.9)
    ylim = (4.1, 5.7)

    #get uniform coordinates
    x_coord = np.random.uniform(low=xlim[0], high=xlim[1], size=(size,))
    y_coord = np.random.uniform(low=ylim[0], high=ylim[1], size=(size,))

    return (x_coord, y_coord)


def create_corridor_scenario(size, free_flow_speed, pedestrian_radius, base_file_path, output_folder_path):
    """use the base scenario file to create a new scenario file variant with defined settings written into the output folder

    :param size: numbere of pedestrians
    :param free_flow_speed: free flow speed of pedestrians
    :param pedestrian_radius: radius of pedestrians
    :param base_file_path: file path of the base scenario
    :param output_folder_path: ouput folder path
    """
    #get coordinates
    (x_coord, y_coord) = get_uniform_pedestrian_coords_for_corridor(size)

    with open(base_file_path, 'r') as base:
        data = base.read()
        obj = json.loads(data)
        #set new name
        obj["name"] += "_" + str(size).zfill(3)
    
        #set pedestrians
        for i in range(len(x_coord)):
            obj["scenario"]["topography"]["dynamicElements"].append({
                "attributes" : {
                    "id" : i+10,
                    "radius" : pedestrian_radius,
                    "densityDependentSpeed" : False,
                    "speedDistributionMean" : 1.34,
                    "speedDistributionStandardDeviation" : 0.26,
                    "minimumSpeed" : 0.5,
                    "maximumSpeed" : 2.2,
                    "acceleration" : 2.0,
                    "footstepHistorySize" : 4,
                    "searchRadius" : 1.0,
                    "walkingDirectionCalculation" : "BY_TARGET_CENTER",
                    "walkingDirectionSameIfAngleLessOrEqual" : 45.0
                },
                "source" : None,
                "targetIds" : [ 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4 ],
                "nextTargetListIndex" : 0,
                "isCurrentTargetAnAgent" : False,
                "position" : {
                    "x" : x_coord[i],
                    "y" : y_coord[i]
                },
                "velocity" : {
                    "x" : 0.0,
                    "y" : 0.0
                },
                "freeFlowSpeed" : free_flow_speed,
                "followers" : [ ],
                "idAsTarget" : -1,
                "isChild" : False,
                "isLikelyInjured" : False,
                "psychologyStatus" : {
                    "mostImportantStimulus" : None,
                    "threatMemory" : {
                    "allThreats" : [ ],
                    "latestThreatUnhandled" : False
                    },
                    "selfCategory" : "TARGET_ORIENTED",
                    "groupMembership" : "OUT_GROUP",
                    "knowledgeBase" : {
                    "knowledge" : [ ]
                    }
                },
                "groupIds" : [ ],
                "groupSizes" : [ ],
                "trajectory" : {
                    "footSteps" : [ ]
                },
                "modelPedestrianMap" : { },
                "type" : "PEDESTRIAN"
            })

    new_scenario_path = output_folder_path + "/corridor_" + str(size).zfill(3) + ".scenario"
    
    #save as new scenario file
    with open(new_scenario_path, 'w') as new:
        json.dump(obj, new, indent=4)