# Load package
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_sk(x_y, xi_yi_array):
        K = xi_yi_array.shape[0]
        # initialize sk as 0
        sk = 0.
        for xi_yi in xi_yi_array:
            sk += np.sqrt(np.sum(np.square(x_y - xi_yi)))
        return sk / K

def generate_data(scenario: str="bottleneck", scenario_specificaton: int=70, data_per_scenario: int=2000, K: int=10):
    scenario_specification_str = str(scenario_specificaton).zfill(3)
    #"vadere_data_raw\\bottleneck\\Bottleneck_070\\pedestrianPosition.txt"
    data_path_position = "vadere_data_raw/" +scenario+"/" +scenario.capitalize()+"_"+scenario_specification_str+"/pedestrianPosition.txt" 
    data_path_velocity = "vadere_data_raw/" +scenario+"/" +scenario.capitalize()+"_"+scenario_specification_str+"/pedestrianSpeedInArea.txt"
    
    raw_data_position = pd.read_csv(data_path_position, sep=" ")
    raw_data_velocity = pd.read_csv(data_path_velocity, sep=" ")

    last_timestep = raw_data_position['timeStep'].iloc[-1]
    last_timestep_enough_pedestrians = np.floor(last_timestep*0.8) # tested empirically

    raw_data_position = raw_data_position[raw_data_position['timeStep'] < last_timestep_enough_pedestrians]
    raw_data_velocity = raw_data_velocity[raw_data_velocity['timeStep'] < last_timestep_enough_pedestrians]

    # print(raw_data_position)
        
    # convert pedestrians in array of tuples id, dataframe of each pedestrian timestep    
    data_array = [pedestrian for pedestrian in raw_data_position.groupby('pedestrianId')]
    # initialize velocities in each coordinate as empty
    pedestrian_velocitiesX = {}
    pedestrian_velocitiesY = {}
    delta_t = 0.4
    for ped_id, pedestrian in data_array:
        # get velocities
        x = pedestrian['x-PID3'].to_numpy()
        y = pedestrian['y-PID3'].to_numpy()
        # initialize velocities
        pedestrian_velocitiesX[ped_id] = np.zeros(len(x))
        pedestrian_velocitiesY[ped_id] = np.zeros(len(y))
        # calculate velocities and save them
        pedestrian_velocitiesX[ped_id][0:len(x)-1] = (x[1:len(x)] - x[0:len(x)-1]) / delta_t
        pedestrian_velocitiesY[ped_id][0:len(y)-1] = (y[1:len(y)] - y[0:len(y)-1]) / delta_t

        # set last value to the previous
        pedestrian_velocitiesX[ped_id][len(x)-1] = pedestrian_velocitiesX[ped_id][len(x)-2]
        pedestrian_velocitiesY[ped_id][len(y)-1] = pedestrian_velocitiesY[ped_id][len(y)-2]



    velocity_measurement_area = raw_data_velocity[raw_data_velocity['speedInAreaUsingAgentVelocity-PID5'] != -1]
    position_measurement_area = raw_data_position[raw_data_velocity['speedInAreaUsingAgentVelocity-PID5'] != -1]

    velocity_measurement_area_sample = velocity_measurement_area.sample(frac=data_per_scenario/velocity_measurement_area.shape[0], random_state = 1).reset_index()
    position_measurement_area_sample = position_measurement_area.sample(frac=data_per_scenario/velocity_measurement_area.shape[0], random_state = 1).reset_index()

    data = position_measurement_area_sample.join(velocity_measurement_area_sample['speedInAreaUsingAgentVelocity-PID5'])


    position_relative_matrix = np.empty(shape=(data_per_scenario, 2*K))
    velocity_relative_matrix = np.empty(shape=(data_per_scenario, 2*K))
    sk_matrix = np.empty(shape=(data_per_scenario, 1))
    assign_rel = np.array([np.array(["x-rel" + str(i+1), "y-rel" + str(i+1)]) for i in range(K)]).flatten()
    assign_rel_velocity = np.array([np.array(["v-rel" + str(i+1), "u-rel" + str(i+1)]) for i in range(K)]).flatten()
    assign_rel_sk = np.array(["s_k"])


    for index, row in data.iterrows(): 
        position_pedestrian = np.reshape([row['x-PID3'], row['y-PID3']], (1,2))
        # get current timestep
        currentTimestep = int(row['timeStep'])
        # get velocities of current pedestrian
        velocity_pedestrian = np.reshape([pedestrian_velocitiesX[int(row['pedestrianId'])][currentTimestep-1], pedestrian_velocitiesY[int(row['pedestrianId'])][currentTimestep-1]], (1,2))
        position_timestep = raw_data_position[raw_data_position['timeStep']==row['timeStep']]
        dis  = cdist(position_pedestrian, np.reshape([position_timestep['x-PID3'], position_timestep['y-PID3']], (2,-1)).T)
        neighbours = dis.argsort()[0, 1:K+1]
        position_relative = position_timestep.loc[:,['x-PID3', 'y-PID3']].to_numpy()[neighbours] - position_pedestrian
        neighbours_timesteps = position_timestep.loc[:,['pedestrianId']].to_numpy()[neighbours]
        # initialize sk
        sk = []
        # calculate sk
        sk.append(calculate_sk(position_pedestrian, position_timestep.loc[:,['x-PID3', 'y-PID3']].to_numpy()[neighbours]))
        # intialize relative positions array
        velocity_relative = []
        # calculate relative velocities
        for [ped_id] in neighbours_timesteps:
            velocity_relative.append(np.reshape([pedestrian_velocitiesX[ped_id][currentTimestep-1], pedestrian_velocitiesY[ped_id][currentTimestep-1]], (1,2)) - velocity_pedestrian)
        # print(type(velocity_relative))
        # print(type(position_relative))
        position_relative = position_relative.flatten()
        velocity_relative = np.array(velocity_relative).flatten()
        sk = np.array(sk).flatten()
        position_relative_matrix[index, :] = position_relative
        velocity_relative_matrix[index, :] = velocity_relative
        sk_matrix[index, :] = sk
       
    position_relative_matrix_df = pd.DataFrame(position_relative_matrix, index = range(data_per_scenario), columns=assign_rel)
    velocity_relative_matrix_df = pd.DataFrame(velocity_relative_matrix, index = range(data_per_scenario), columns=assign_rel_velocity)
    sk_matrix_df = pd.DataFrame(sk_matrix, index = range(data_per_scenario), columns=assign_rel_sk)

    data = data.join(position_relative_matrix_df[assign_rel])
    data = data.join(velocity_relative_matrix_df[assign_rel_velocity])
    data = data.join(sk_matrix_df[assign_rel_sk])
    
    return data