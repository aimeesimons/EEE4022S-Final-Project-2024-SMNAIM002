import powerfactory as pf
import csv
import random

app = pf.GetApplication()
app.ResetCalculation()
app.ClearOutputWindow()
comInc = app.GetFromStudyCase("ComInc")
comSim = app.GetFromStudyCase("ComSim")
res = app.GetFromStudyCase('*.ElmRes')

def prepare_dynamic_sim(monitored_variables, sim_type = 'ins', start_time=0.0, step_size=0.0005, end_time=1):
    res = app.GetFromStudyCase('*.ElmRes')
    for elm_name, var_names in monitored_variables.items():
        elements = app.GetCalcRelevantObjects(elm_name)
    for element in elements:
        res.AddVars(element, *var_names)
    comInc = app.GetFromStudyCase("ComInc")
    comSim = app.GetFromStudyCase("ComSim")
    comInc.iopt_sim = sim_type
    comInc.tstart = start_time
    comInc.dtgrd = step_size
    comSim.tstop = end_time

    comInc.Execute()
def run_dynamic_sim():
    return bool(comSim.Execute())
def get_dynamic_results(elm_name,var_name):
    element = app.GetCalcRelevantObjects(elm_name)[0]
    app.ResLoadData(res)
    col_index = app.ResGetIndex(res, element, var_name)
    n_rows = app.ResGetValueCount(res, 0)
    time = []
    var_values = []
    for i in range(n_rows):
        time.append(app.ResGetData(res,i, -1)[1])
        var_values.append(app.ResGetData(res,i,col_index)[1])
    return time, var_values

MONITORED_VARIABLES = {
'Central Load.ElmLod': ['m:I:bus1:A','m:I:bus1:B','m:I:bus1:C','n:U:bus1:A','n:U:bus1:B','n:U:bus1:C']
}

for i in range(20000):
    app.ResetCalculation() 
    app.ClearOutputWindow() 
    start_time = random.random()
    prepare_dynamic_sim(monitored_variables=MONITORED_VARIABLES, start_time=start_time, end_time=start_time+0.5)
    run_dynamic_sim()
    t, I_A = get_dynamic_results('Central Load.ElmLod','m:I:bus1:A')
    _, I_B = get_dynamic_results('Central Load.ElmLod','m:I:bus1:B')
    _, I_C = get_dynamic_results('Central Load.ElmLod','m:I:bus1:C')

    _, V_A = get_dynamic_results('Central Load.ElmLod','n:U:bus1:A')
    _, V_B = get_dynamic_results('Central Load.ElmLod','n:U:bus1:B')
    _, V_C = get_dynamic_results('Central Load.ElmLod','n:U:bus1:C')


    with open(f'non_faulty{i}.csv','w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['t','I_A','I_B','I_C', 'V_A', 'V_B', 'V_C'])
        for row in zip(t, I_A, I_B, I_C, V_A, V_B, V_C):
            csvwriter.writerow(row)

