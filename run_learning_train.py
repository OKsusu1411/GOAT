import sys
import os
import runpy

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_path = os.path.join(project_root, 'Simulation', 'Tasks', 'GOAT_PD_stand', 'train.py')

runpy.run_path(script_path, run_name='__main__')