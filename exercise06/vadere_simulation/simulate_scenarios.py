import subprocess
import glob, os


def simulate_scenarios_from_folder(folder_path, output_folder_path, print_output=False):
    """function to run a simulation using the console version of vadere

    :param folder_path: folder path of the scenario files to be simulated
    :param output_folder_path: output folder path of the simulation output files
    :param print_output: parameter whether to print the ouput from std:out, defaults to False
    """
    #get all scenario files in folder
    file_path = os.path.dirname(__file__)
    os.chdir(file_path + '/../' + folder_path)

    for file in glob.glob("*.scenario"):
        #if not base file simulate it
        if file != 'corridor.scenario' and file != 'bottleneck.scenario':
            print(f"Simulate Scenario File: {file}...")
            os.chdir(file_path)
            command = f"java -jar vadere-console.jar scenario-run --scenario-file ../{folder_path}{file} --output-dir=../{output_folder_path}"
            result = subprocess.check_output([command], shell=True)
            if print_output:
                print(result)