import sys
import os
import runpy

# 프로젝트 루트 디렉토리를 Python 경로에 추가하여 모듈 임포트 문제를 해결합니다.
# 이를 통해 하위 디렉토리의 스크립트 내에서 'lib' 모듈을 찾을 수 있습니다.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 실행해야 할 스크립트의 경로를 지정합니다.
# script_path = os.path.join(project_root, 'lib', 'low_level_controller', 'pd_controller_haechan.py')
script_path = os.path.join(project_root, 'lib', 'low_level_controller', 'joint_data_collector.py')
# script_path = os.path.join(project_root, 'Simulation', 'Tasks', 'GOAT_PD_stand', 'train.py')
# 스크립트를 직접 실행한 것처럼 실행합니다.
# 'run_name="__main__"' 인자는 대상 스크립트의
# if __name__ == "__main__": 블록이 실행되도록 보장합니다.
runpy.run_path(script_path, run_name='__main__')