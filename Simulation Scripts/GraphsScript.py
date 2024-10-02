import powerfactory as pf
import csv
import random
 
# Establish connection to PowerFactory application
app = pf.GetApplication()
app.ResetCalculation()  # Reset any previous calculations
app.ClearOutputWindow()  # Clear the output window for a clean start
res = app.GetCalcRelevantObjects("*.ElmRes")

# Define the function to initiate a short circuit event
def initiate_short_circuit(line_name, fault_location=0.5, fault_type=0, phase = None, start_time=0.0, step_size=0.0005, end_time=0.08):
    """
    Initiates a short circuit event on a specified transmission line.
 
    :param line_name: Name of the transmission line where the fault will be applied.
    :param fault_location: The relative location of the fault on the line (0 = start, 1 = end, 0.5 = middle).
    :param fault_type: The type of fault ('3ph' for three-phase, '1ph' for single-phase, etc.).
    :param start_time: The start time of the simulation.
    :param step_size: The time step size for the simulation.
    :param end_time: The end time of the simulation.
    """
 
    # Retrieve the line object from the study case
    line = app.GetCalcRelevantObjects(line_name)[0]
    if not line:
        raise ValueError(f"Transmission line {line_name} not found.")
 
    # Create a fault object in the study case
    fault = app.GetFromStudyCase("IntEvt")
    event = fault.CreateObject('EvtShc', 'ShortCircuit')
    events = fault.GetContents()[0]
    event.p_target = line
    event.time = 0.02
    #fault.SetAttribute('loc_name', f"Fault on {line_name}")
    # Set the fault to occur on the specified line
    #fault.SetAttribute(, line) # Attach the fault to the line
    #event.shcLocation = fault_location  # Location of the fault on the line
    event.R_f = 0.1
    event.X_f = 0.0
    #fault.SetAttribute('Fault Type', fault_type)
    event.i_shc = fault_type
    match (fault_type):
        case 1: event.i_p2psc = phase
        case 2: event.i_pspgf = phase
        case 3: event.i_p2pgf = phase
    # Retrieve simulation control objects from the study case
    comInc = app.GetFromStudyCase("ComInc")
    comSim = app.GetFromStudyCase("ComSim")
 
    # Configure simulation parameters
    comInc.iopt_sim = 'ins'  # Use 'rms' for RMS value calculation, or use another type as needed
    comInc.tstart = start_time
    comInc.dtemt = step_size
    comSim.tstop = end_time
 
    # Execute the initial setup of the simulation
    comInc.Execute()
 
    # Run the simulation
    sim_result = comSim.Execute()
    #if not sim_result:
     #   raise RuntimeError("Simulation failed to execute.")
 
    print(f"Short circuit event initiated on {line_name} at location {fault_location} with fault type {fault_type}.")
 
    return fault  # Return the fault object for potential further use
 
# Function to extract and save simulation results to a CSV file
def save_simulation_results(monitored_variables, output_file):
    """
    Extracts the simulation results for the specified monitored variables and saves them to a CSV file.
 
    :param line_name: Name of the line to extract results from.
    :param monitored_variables: Dictionary with element names as keys and list of variable names as values.
    :param output_file: The path to the output CSV file.
    """
    desired_buses = ['Central-NE S1/BB1', 'Central-NE S2/BB1', 'Central-NE S3/BB1', 'NE-Eastern S1/BB1', 'NW-Central S1/BB1', 'SE-Central S1/BB1', 'SE-Central S2/BB1', 'SE-NW S1/BB1', 'SE-NW S2/BB1', 'SE-NW S3/BB1','Southern-Central S1/BB1', 'Southern-Eastern S1/BB1', 'Southern-Eastern S2/BB1', 'Southern-Eastern S3/BB1', 'Southern-Eastern S4/BB1', 'Central/BB2','Eastern/BB2','NE/BB2','NW/BB2','SW/BB2','Southern/BB2']
    time_series = {}
    res = app.GetFromStudyCase('*.ElmRes')
    for elm_name, var_names in monitored_variables.items():
        if '*' in elm_name:
          busbars = app.GetCalcRelevantObjects(elm_name)
          app.ResLoadData(res)
          for busbar in busbars:
            for bus in desired_buses:
              if bus in busbar.GetFullName():
                  for var_name in var_names:
                      col_index = app.ResGetIndex(res, busbar, var_name)
                      n_rows = app.ResGetValueCount(res, 0)
                      time_series['t'] = [app.ResGetData(res, i, -1)[1] for i in range(n_rows)]
                      time_series[busbar.GetFullName()[busbar.GetFullName().index("ElmSubstat\\")+11:] + ' ' + var_name] = [app.ResGetData(res, i, col_index)[1] for i in range(n_rows)]
        else:
         element = app.GetCalcRelevantObjects(elm_name)[0]
         app.ResLoadData(res)  # Load simulation results into memory
         for var_name in var_names:
           col_index = app.ResGetIndex(res, element, var_name)
           n_rows = app.ResGetValueCount(res, 0)
           time_series['t'] = [app.ResGetData(res, i, -1)[1] for i in range(n_rows)]
           time_series[elm_name[:elm_name.index('.')] + ' ' + var_name] = [app.ResGetData(res, i, col_index)[1] for i in range(n_rows)]
 
    # Save the extracted results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['t'] + list(time_series.keys())[1:])  # Write header
        for i in range(len(time_series['t'])):
            csvwriter.writerow([time_series['t'][i]] + [time_series[var][i] for var in time_series.keys() if var != 't'])
 
    print(f"Simulation results saved to {output_file}")
#def clearSimEvents(faultFolder):
 # faultFolder = app.GetFromStudyCase("*.IntEvt")
  #cont = faultFolder.GetContents()
  #for obj in cont:
   # obj.DeleteObject()
# Example Usage
MONITORED_VARIABLES = {
'*.ElmTerm': ['m:Ishc:A','m:Ishc:B','m:Ishc:C','m:U:A','m:U:B','m:U:C'],
'Central - Southern 400kV.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Central - Southern 400kV S1.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Eastern - NE 400kV 1.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Eastern - NE 400kV 1a.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Eastern - NE 400kV 2.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Eastern - NE 400kV 2a.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NE - Central 400kV.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NE - Central 400kV S1.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NE - Central 400kV_b.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NE - Central 400kV_c.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NW - Central 400kV.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'NW - Central 400kV S1.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - Central 400kV.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - Central 400kV S2.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - Central 400kV_a.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - NW 400kV.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - NW 400kV_a.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - NW 400kV_b.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'SE - NW 400kV_c.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Southern - Eastern 400kV_S5.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Southern - Eastern 400kV_a.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Southern - Eastern 400kV_b.ElmLne':['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'], 
'Southern - Eastern 400kV_c.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
'Southern - Eastern 400kV_d.ElmLne': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C'],
}
faults = {
    (0, None) : 'ABC',
    (1, 0): 'AB',
    (1, 1): 'BC',
    (1, 2): 'AC',
    (2, 0): 'AG',
    (2, 1): 'BG',
    (2, 2): 'CG',
    (3, 0): 'ABG',
    (3, 1): 'BCG',
    (3, 2): 'ACG'
}
locations = [i for i in range(1,100)]
faultOn = 'Southern/BB2.ElmTerm'
bus = 'Southern_BB2'
# Reset the application before starting a new simulation

# Initiate the short circuit on a specific line
#faultFolder = app.GetFromStudyCase("IntEvt")
#event = faultFolder.GetContents("*.EvtShc")[0]
#faultFolder.SetAttribute('loc', 0.2)
j = 0
for (key1, key2), value in faults.items():
   #start_time = random.random()*0.01
   app.ResetCalculation()
   app.ClearOutputWindow() 
   # fault = app.GetFromStudyCase("IntEvt")
   # events = fault.GetContents()[0]
   # events.shcLocation = loc
   fault = initiate_short_circuit(f'{faultOn}', fault_location=None, fault_type=key1, phase=key2)
   # Extract and save the simulation results
   output = f'Model\\Fault Localisation\\Graphs\\Graph_{j}_{bus}.csv'
   save_simulation_results(MONITORED_VARIABLES, output)
   # Clean up (delete) the fault object after the simulation, if necessary
   fault.Delete()
   j+=1
 
print("Simulation and data extraction complete.")