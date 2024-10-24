import powerfactory as pf
import csv
import random
import pandas as pd
import numpy as np
import pywt
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import joblib
from tkinter import messagebox
import time
import pickle
import tensorflow as tf
import tensorflow.keras
import sys
import io
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import re


# Backup the original stdout
original_stdout = sys.stdout

# Redirect stdout to suppress output
sys.stdout = io.StringIO()

# Establish connection to PowerFactory application
app = pf.GetApplication()
app.ResetCalculation()  # Reset any previous calculations
app.ClearOutputWindow()  # Clear the output window for a clean start
res = app.GetCalcRelevantObjects("*.ElmRes")

class NodeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, fc_hidden_size):
        super(NodeGCN, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu = nn.ReLU()

        self.fc =  nn.Linear(32*120, in_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Fully connected layers
        self.fc1 = nn.Linear(out_channels, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, out_channels) 

    def forward(self, x, edge_index, batch):
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

      
        out = global_mean_pool(x, batch)

        x = F.relu(self.fc1(out))
        x = self.fc2(x)

        return x

class EdgeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, fc_hidden_size):
        super(EdgeGCN, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu = nn.ReLU()

        self.fc =  nn.Linear(32*120, in_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Fully connected layers
        self.fc1 = nn.Linear(out_channels, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, out_channels)  

    def forward(self, edge_attr, edge_index, batch):
        x = self.conv1d_1(edge_attr)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

      
        row, col = edge_index
        if batch is None:
          # Handle single graph case
          out = torch.zeros((1, x.size(1)), device=x.device)
          out.index_add_(0, torch.zeros(row.size(0), dtype=torch.long, device=row.device), x)
        else:
          out = torch.zeros((batch.max().item() + 1, x.size(1)), device=x.device)
          out.index_add_(0, batch[row], x)

        x = F.relu(self.fc1(out))
        x = self.fc2(x)

        return x

# Define the function to initiate a short circuit event
def initiate_short_circuit(line_name, fault_location=0.5, fault_type=0, phase = None, start_time=0.0, step_size=0.0005, end_time=0.5, initiate=True):
    """
    Initiates a short circuit event on a specified transmission line.
 
    :param line_name: Name of the transmission line where the fault will be applied.
    :param fault_location: The relative location of the fault on the line (0 = start, 1 = end, 0.5 = middle).
    :param fault_type: The type of fault ('3ph' for three-phase, '1ph' for single-phase, etc.).
    :param start_time: The start time of the simulation.
    :param step_size: The time step size for the simulation.
    :param end_time: The end time of the simulation.
    """
    
    if initiate:
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
        if fault_location is not None:
          events.shcLocation = fault_location  # Location of the fault on the line
        event.R_f = 0.1
        event.X_f = 0.0
        #fault.SetAttribute('Fault Type', fault_type)
        event.i_shc = fault_type
        match (fault_type):
            case 1: event.i_p2psc = phase
            case 2: event.i_pspgf = phase
            case 3: event.i_p2pgf = phase
    else:
      fault = None
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
        #raise RuntimeError("Simulation failed to execute.")
 
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
 
    time_series = {}
    res = app.GetFromStudyCase('*.ElmRes')
    for elm_name, var_names in monitored_variables.items():
        element = app.GetCalcRelevantObjects(elm_name)[0]
        app.ResLoadData(res)  # Load simulation results into memory
        for var_name in var_names:
            col_index = app.ResGetIndex(res, element, var_name)
            n_rows = app.ResGetValueCount(res, 0)
            time_series['t'] = [app.ResGetData(res, i, -1)[1] for i in range(n_rows)]
            time_series[var_name] = [app.ResGetData(res, i, col_index)[1] for i in range(n_rows)]
 
    # Save the extracted results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['t'] + list(time_series.keys())[1:])  # Write header
        for i in range(len(time_series['t'])):
            csvwriter.writerow([time_series['t'][i]] + [time_series[var][i] for var in time_series.keys() if var != 't'])
 
    print(f"Simulation results saved to {output_file}")

def save_simulation_results_loc(monitored_variables, output_file):
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

def compute_wavelet(signal, i ,j,scales,data_cwt, start,         #compute wavelet for detection and classification
                 waveletname = 'morl'):
                   
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)
    coefficients = coefficients[:127,start:start+127]
    data_cwt[j,:,:,i] = np.abs(coefficients)

def compute_wavelet_loc(signal, i ,j,scales, data_cwt,       #compute wavelet for localisation
                 waveletname = 'morl'):
    
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)
    coefficients = coefficients[:40,:40]
    data_cwt[j,:,:,i] = np.abs(coefficients)

measured_From = {           #dictionary of lines and their reference busss
'Central - Southern 400kV':'Busbar Central\BB2',
'Central - Southern 400kV S1':'Busbar Southern-Central S1\BB1' ,
'Eastern - NE 400kV 1':'Busbar NE- Eastern S1\BB1',
'Eastern - NE 400kV 1a':'Busbar Eastern\BB1',
'Eastern - NE 400kV 2':'Busbar NE - Eastern S1\BB1' ,
'Eastern - NE 400kV 2a':'Busbar Eastern\BB1' ,
'NE - Central 400kV':'Busbar NE\BB2' ,
'NE - Central 400kV S1':'Busbar Central-NE S1\BB1' ,
'NE - Central 400kV_b': 'Busbar Central-NE S2\BB1',
'NE - Central 400kV_c':'Busbar Central-NE S3\BB1' ,
'NW - Central 400kV':'Busbar NW\BB2' ,
'NW - Central 400kV S1':'Busbar NW-Central S1\BB1' ,
'SE - Central 400kV': 'Busbar SW\BB1',
'SE - Central 400kV S2':'Busbar SE-Central S2\BB1',
'SE - Central 400kV_a':'Busbar SE-Central S1\BB1' ,
'SE - NW 400kV':'Busbar NW\BB2' ,
'SE - NW 400kV_a':'Busbar SE-NW S1\BB1' ,
'SE - NW 400kV_b':'Busbar SE-NW S2\BB1' ,
'SE - NW 400kV_c':'Busbar SE-NW S3\BB1' ,
'Southern - Eastern 400kV_S5':'Busbar Southern-Eastern S4\BB1' ,
'Southern - Eastern 400kV_a':'Busbar Southern-Central S1\BB1' ,
'Southern - Eastern 400kV_b':'Busbar Southern-Eastern S1\BB1', 
'Southern - Eastern 400kV_c':'Busbar Southern-Eastern S2\BB1' ,
'Southern - Eastern 400kV_d':'Busbar Southern-Eastern S3\BB1' ,
}
def locate_line(line):
   app.PrintPlain(snr)
   X_input = np.ndarray(shape=(1, 40, 40, 6), dtype=np.float32)
   monitored_variables = {
    f'{line}.ElmLne': ['n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C','m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C'] 
   }
   output = f'Results\\Integration\\Line.csv'
   save_simulation_results(monitored_variables, output)
   df = pd.read_csv(output)
   start_index = df[df['t'] == 0.02].index.min()
   app.PrintPlain(start_index)
   start_index = start_index + 2
   filtered_df = df.iloc[start_index:start_index+40]
   filtered_df = filtered_df.drop('t', axis=1)
   
   i = 0
   j = 0
   for (columnName,columnData) in filtered_df.items():
       input_signal = columnData.values
       factor = 400/max(input_signal)
       input_signal = input_signal*factor
      if snr == 0:
          filtered_signal = input_signal/factor
      else:
          signal_original = add_all_noises(input_signal/5,noise_factors,snr) 
          signal_original = signal_original/factor
          b,a = scipy.signal.cheby1(9, 1, 100, fs=2000, btype='lowpass')
          filtered_signal = scipy.signal.filtfilt(b, a, signal_original)
      signal_normalised = preprocessing.normalize([filtered_signal])
      compute_wavelet_loc(signal_normalised.reshape(-1),i,j,scales=np.arange(1,41),data_cwt=X_input)
      plt.imshow(X_input[0,:,:,i])
      plt.show()
      i += 1
      
   line1 =  re.sub(r'[\s-]+', '_', line)
   model = tf.keras.models.load_model(f"LineModels_new\\{line1}\\{line1}_10.keras")
   location = model.predict(X_input)
   app.PrintPlain(f"Fault is located on {line} at {location[0][0]} km from {measured_From[line]}")
   
   
   
   
   
def extract_graph(): #Extract graph attributes from dataframe
   app.PrintPlain(snr)
   waveforms = []
   graphs = []
   X_graphs = []
   count_nodes = 0
   output = f'Results\\Integration\\Graph.csv'
   save_simulation_results_loc(MONITORED_VARIABLES_1, output)
   graph = pd.read_csv(output)
   start_index = graph[graph['t'] == 0.02].index.min()
   app.PrintPlain(start_index)
   start_index = start_index + 2
   filtered_df = graph.iloc[start_index:start_index+120]
   filtered_df = filtered_df.drop('t', axis=1)
   i = 0
   for (columnName, columnData) in filtered_df.items():
       input_signal = columnData.values
       if max(input_signal) != 0:
           factor = 400/max(input_signal)
           input_signal = input_signal/factor
       else:
           factor = 1
     if snr == 0:
        filtered_signal = input_signal/factor
     else:
        signal_original = add_all_noises(columnData.values/5,noise_factors,snr)  
         signal_original = signal_original/factor
        b,a = scipy.signal.cheby1(9, 1, 100, fs=2000, btype='lowpass')
        filtered_signal = scipy.signal.filtfilt(b, a, signal_original)
     signal_normalised = preprocessing.normalize([filtered_signal])
     waveforms.append(signal_normalised.reshape(-1))
     i+=1
     if i==6:
       graphs.append(waveforms)
       waveforms = []
       i = 0
       count_nodes+=1
     if count_nodes==21:
       X_graphs.append(graphs)
       count_nodes+=1
       graphs=[]
     if count_nodes==46:
       X_graphs.append(graphs)
       graphs = []
       count_nodes = 0
   return X_graphs

def create_graph(graph):  #creating graph object
  x = torch.tensor(graph[0])
  edge_index = torch.tensor([
    [0, 16, 5, 6, 5, 6, 5, 1, 2, 3, 7, 8, 14, 10, 9, 7, 11, 12, 13, 20, 16, 17, 18, 19],
    [16, 15, 6, 4, 6, 4, 3, 0, 1, 2, 8, 0, 9, 0, 10, 13, 14, 11, 12, 4, 17, 18, 19, 20]
    ], dtype=torch.long)
  edge_attr = torch.tensor(graph[1])
  data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
  
  return data

  
def prediction(data, modelNode, modelEdge):
    modelNode.eval()
    node_out = modelNode(data.x.float(), data.edge_index, data.batch)
    modelEdge.eval()
    edge_out = modelEdge(data.edge_attr.float(), data.edge_index, data.batch)
    node_preds = node_out.argmax(dim=1)  # Get predicted class labels for nodes
    edge_preds = edge_out.argmax(dim=1)  # Get predicted class labels for edges 
    app.PrintPlain(node_preds)
    app.PrintPlain(edge_preds)
    
    return node_preds, edge_preds 

# Thermal Noise (Johnson Noise)
import numpy as np

def thermal_noise(signal, noise_power=0.01):
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal


# Electromagnetic Interference
def electromagnetic_interference(signal, freq=50, amplitude=10):
    t = np.linspace(0, len(signal)/1000, len(signal))
    emi = amplitude * np.sin(2 * np.pi * freq * t)
    noisy_signal = signal + emi
    return noisy_signal


# harmonics 
def harmonics(signal, base_freq=50, harmonics_order=[3, 5], amplitudes=[50, -50]):
    t = np.linspace(0, len(signal)/1000, len(signal))
    harmonic_signal = np.zeros_like(signal)
    for order, amplitude in zip(harmonics_order, amplitudes):
        harmonic_signal += amplitude * np.sin(2 * np.pi * order * base_freq * t)
    return signal + harmonic_signal


# Switching noise
def switching_noise(signal, noise_amplitude=0.5, switch_count=5):
    noisy_signal = signal.copy()
    for _ in range(switch_count):
        idx = np.random.randint(0, len(signal))
        noisy_signal[idx] += noise_amplitude
    return noisy_signal

# Arc Noise
def arc_noise(signal, burst_amplitude=5, burst_duration=5, burst_count=3):
    noisy_signal = signal.copy()
    for _ in range(burst_count):
        start_idx = np.random.randint(0, len(signal) - burst_duration)
        burst = burst_amplitude * np.random.randn(burst_duration)
        noisy_signal[start_idx:start_idx + burst_duration] += burst
    return noisy_signal


# Flicker Noise (1/f or pink noise)
def flicker_noise(signal, beta=1):
    f = np.fft.rfftfreq(len(signal))
    f[0] = 1  # To avoid division by zero at DC component
    flicker = (1 / f**beta) * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
    flicker_signal = np.fft.irfft(flicker)
    return signal + flicker_signal[:len(signal)]


# Impulse Noise (Bust Noise)
def impulse_noise(signal, impulse_amplitude=2.5, impulse_count=10):
    noisy_signal = signal.copy()
    for _ in range(impulse_count):
        idx = np.random.randint(0, len(signal))
        noisy_signal[idx] += impulse_amplitude
    return noisy_signal


# Corona Noise
def corona_noise(signal, freq=10000, amplitude=0.05):
    t = np.linspace(0, len(signal)/1000, len(signal))
    corona = amplitude * np.sin(2 * np.pi * freq * t)
    noisy_signal = signal + corona
    return noisy_signal

def add_all_noises(signal, noise_factors, snr_db):
    """
    Adds different types of noise to the signal with varying strengths.
    :param signal: Original signal
    :param noise_factors: Dictionary of noise types and their corresponding scaling factors
    :param snr_db target signal snr
    :return: Signal with added noise
    """
    noisy_signal = signal.copy()
    
    # Add thermal noise
    noisy_signal += noise_factors['thermal']() * thermal_noise(signal, np.mean(signal**2)/10**(snr_db/10))
    
    # Add electromagnetic interference
    noisy_signal += noise_factors['emi']() * electromagnetic_interference(signal, freq=60, amplitude=0.5)
    
    # Add harmonic noise
    noisy_signal += noise_factors['harmonics']() * harmonics(signal)
    
    # Add switching noise
    noisy_signal += noise_factors['switching']() * switching_noise(signal)
    
    noisy_signal += noise_factors['arc']() * arc_noise(signal)

    noisy_signal += noise_factors['impulse']() * impulse_noise(signal)

    noisy_signal += noise_factors['flicker']() * flicker_noise(signal)

    noisy_signal += noise_factors['corona']() * corona_noise(signal)
    
    return noisy_signal

noise_factors = {
    'thermal': lambda: random.random(),      
    'emi': lambda: random.random(),         
    'harmonics': lambda: random.random(),   
    'switching': lambda: random.random(),    
    'arc': lambda: random.random(),
    'flicker': lambda: random.random(),
    'impulse': lambda: random.random(),
    'corona': lambda: random.random()
}



MONITORED_VARIABLES = {
    'Central Load.ElmLod': ['m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C', 'n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C']
}

MONITORED_VARIABLES_1 = {
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

locations = [i for i in range(101)] #can be used if needed to loop through all locations

snr_levels = [0,20,30] 

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
with open("ordinal_encoder.pickle", 'rb') as f:
  ordinal_encoder = pickle.load(f)
with open("label_encoder.pickle", 'rb') as f:
  label_encoder = pickle.load(f)
    
X_input = np.ndarray(shape=(1, 127, 127, 6), dtype=np.float32)

random_num = random.randint(0,9)

(key1, key2), value = list(faults.items())[random_num] #choosing a random fault type

start_time = 0
total_time = 0
app.ResetCalculation()
app.ClearOutputWindow() 
initiate = True
if initiate == False:
  value = 'No'
loc = random.randint(0,100)
fault = initiate_short_circuit('SE - NW 400kV_a.ElmLne',fault_location=loc, fault_type=key1, phase=key2, start_time=start_time, end_time=start_time + 0.5, initiate=initiate)
# Extract and save the simulation results
output = f'Results\\Integration\\{value}_fault.csv'
save_simulation_results(MONITORED_VARIABLES, output)


snr = np.random.choice(snr_levels) #choose a random snr level
j = 0
i = 0
X_input_csv = pd.read_csv(output)
X_input_csv = X_input_csv[['n:U:bus1:A', 'n:U:bus1:B', 'n:U:bus1:C','m:I:bus1:A', 'm:I:bus1:B', 'm:I:bus1:C']]
start_time = time.time()
start = int(random.random()*0.1*2000)
for (columnName,columnData) in X_input_csv.items():
    input_signal = columnData[:1000].values
    if max(input_signal) <1:
        input_signal = input_signal*666
    if snr == 0:
        filtered_signal = input_signal
    else:
        signal_original = add_all_noises(input_signal/5,noise_factors,snr) 
        if "I" in columnName:
            signal_original = signal_original/666
        b,a = scipy.signal.cheby1(9, 1, 100, fs=2000, btype='lowpass')
        filtered_signal = scipy.signal.filtfilt(b, a, signal_original)
    signal_normalised = preprocessing.normalize([filtered_signal])
    compute_wavelet(signal_normalised.reshape(-1),i,j,scales=np.arange(1,128),data_cwt=X_input,start=start)
    i += 1
end_time = time.time()
app.PrintPlain(snr)
app.PrintPlain(f"Feature Extraction conducted in {end_time-start_time}s")#messagebox.showinfo("Feature Extraction", f"Feature Extraction conducted in {end_time-start_time}s")
total_time += end_time-start_time
## Detection
start_time = time.time()
detect = joblib.load("DetectionModel_mixed_noise_2.joblib") #accessing detection model
pred = detect.predict(X_input)
pred_encoded = (pred > 0.5).astype(int)
pred_class = ordinal_encoder.inverse_transform(pred_encoded.reshape(-1,1))
end_time = time.time()
total_time += end_time-start_time
if pred_class=='Fault Detected':
   app.PrintPlain(f"Fault Detected in {end_time-start_time}s.")#messagebox.showinfo("Fault Detected", f"Fault Detected in {end_time-start_time}s.")
   start_time = time.time()
   classify = joblib.load("ClassificationModel_mixed_noise_2.joblib") #accessing the classification model
   pred = classify.predict(X_input)
   pred_encoded = (pred > 0.5).astype(int)
   pred_indices = np.argmax(pred_encoded, axis=1)
   pred_class = label_encoder.inverse_transform(pred_indices)
   end_time = time.time()
   total_time += end_time-start_time
   if pred_class[0] == 'NNNN':
        app.PrintPlain("No Fault Detected")#messagebox.showinfo("Classification", "No Fault Detected")
   else:
        app.PrintPlain(f"{pred_class[0]} fault detected in {end_time-start_time}s")#messagebox.showinfo("Fault Classified", f"{pred_class} fault detected in {end_time-start_time}s")
        # go to location algorithm
        with open("graph_nodes_ordinal_encoder.pkl",'rb') as f:
             ordinal_encoder_node = pickle.load(f)
        with open("graph_edges_ordinal_encoder.pkl",'rb') as f:
             ordinal_encoder_edge = pickle.load(f)
        start_time = time.time()
        graph = extract_graph()
        app.PrintPlain("Extracted Graph")
        data = create_graph(graph)
        app.PrintPlain("Create Data Object")
        node_preds, edge_preds = prediction(data, torch.load("model_node_2.pth"), torch.load("model_edge_2.pth")) #predicting the nodes and edges
        node_preds = ordinal_encoder_node.inverse_transform(node_preds.reshape(-1,1)) 
        edge_preds = ordinal_encoder_edge.inverse_transform(edge_preds.reshape(-1,1))
        app.PrintPlain(node_preds)
        app.PrintPlain(edge_preds)
        #locate_line("Central - Southern 400kV S1")
        if node_preds[0][0] == 'Line':
            line = edge_preds[0][0]
            if line=='SE - NW 400kV_a': #this can be replaced with the selected line
              locate_line(line)
            else:
              locate_line("SE - NW 400kV_a") #if the model guesses incorrectly
        elif edge_preds[0][0]=='Node':
         app.PrintPlain(f'Fault occurred on Bus {node_preds[0][0]}')  
        end_time = time.time()
        app.PrintPlain(f"Location algorithm completed in {end_time-start_time}s")
        total_time += end_time-start_time
        app.PrintPlain(f"Total Time for completion: {total_time}s")
        fault.Delete()
else:
    app.PrintPlain("No Fault Detected")#messagebox.showinfo("Detection", "No Fault Detected")
    if fault is not None:
       fault.Delete()


     
     

  
