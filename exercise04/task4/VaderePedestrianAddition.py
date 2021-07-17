import json
import os

def add_pedestrians_scenario(x, y):
    current_dir = os.path.dirname(__file__)
    rel_path = "scenarios\\rimea_06_corner_self_osm.scenario"
    # get the path to scenario file
    path = os.path.join(current_dir, rel_path)
    # open file for reading and change obtained JSON file
    f = open(path, "r")
    data = json.load(f)
    data['name'] = "rimea_06_corner_self_osm_with_pedestrian"
    # set new point
    for i in range(len(x)):
      data["scenario"]["topography"]["dynamicElements"].append({
      "attributes" : {
        "id" : i+2,
        "radius" : 1,
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
      "targetIds" : [ 1 ],
      "nextTargetListIndex" : 0,
      "isCurrentTargetAnAgent" : False,
      "position" : {
        "x" : x[i],
        "y" : y[i]
      },
      "velocity" : {
        "x" : 0.0,
        "y" : 0.0
      },
      "freeFlowSpeed" : 1.327098490577325,
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

    f.close()

    new_rel_path = "scenarios\\rimea_06_corner_self_osm_with_pedestrian.scenario"
    # get path for new updated corner scenario which need to be created
    new_path = os.path.join(current_dir, new_rel_path)
    try:
        fi = open(new_path, "x")
    except:
        fi = open(new_path, "w")
    fi.write(json.dumps(data, indent=4))
    fi.close()

# read_scenario()