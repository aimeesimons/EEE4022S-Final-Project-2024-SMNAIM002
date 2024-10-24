## These are the files utilised for the automated simulations

The scripts include:
- GraphsScript.py -> which simualtes a fault on a transmission line and collects all the data from the buses and lines
- IntegrationTest.py -> which combines all the fault identification algorithms
- NoFaultscript.py -> which simulates a normal EMT simualtion, with no faults
- shortcircuitScript.py -> which simulates faults along transmission lines and either collects from the Central Load or the selected transmission line.

To use these scripts:
1. Open DigSilent DataManger
2. Select the current project
3. Proceed to Libraries -> Scripts
4. Create a Python Script
5. Copy the scripts above exactly and then paste in DigSilent's Embedded Python Environment
6. Execute

*NB The Graph and Integration Script might not work for just any copy of the 21-Bus system. It will only work with the model specified in * DigSilent Models/ * 
