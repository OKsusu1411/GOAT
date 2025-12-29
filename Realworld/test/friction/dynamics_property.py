import pinocchio as pin
import numpy as np
from ..lib.env.GOAT_base_env_cfg import GOAT_ASSET

class DynamicsProperty:
    def __init__(self):
        """
        로봇의 동역학적 속성을 저장하기 위한 클래스 초기화
        """
        self.urdf_path = GOAT_ASSET["urdf_path"]
        self.model = None
        
        # RNEA 계산에 필요한 속성을 저장할 리스트
        # 각 리스트의 인덱스는 model의 조인트/링크 인덱스(1부터 시작)와 일치합니다.
        self.link_properties = []  # 각 링크의 [질량, 질량중심, 관성텐서] 딕셔너리를 저장
        self.screw_axes = []       # 각 조인트의 [스크류 축] numpy 배열을 저장

    def extract_robot_model_info(self):
        """
        URDF 파일에서 로봇의 동역학적 파라미터를 추출하여 인스턴스 변수에 저장합니다.
        RNEA 알고리즘에 사용하기 적합한 형태로 데이터를 구성합니다.
        """
        # URDF 파일로부터 로봇 모델 빌드
        try:
            # Pinocchio는 부동 소수점 기반 조인트를 기본으로 모델을 빌드합니다.
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            print("URDF load success!!")
        except Exception as e:
            print(f"URDF load failed!!: {e}")
            return

        print(f"\n===== Robot Model Info: {self.model.name} =====")
        print(f"Total joints (including universe): {self.model.njoints}")
        print(f"Degrees of Freedom (DoF): {self.model.nv}")

        # 'universe' 조인트(인덱스 0)를 제외하고 실제 움직이는 조인트/링크에 대한 정보를 추출
        for i in range(1, self.model.njoints):
            # 1. 링크의 동역학적 속성 (질량, 질량중심, 관성텐서) 저장
            # model.inertias[i]는 i번째 조인트에 의해 움직이는 자식 링크의 관성 객체(pin.Inertia)입니다.
            inertia_obj = self.model.inertias[i]
            
            link_data = {
                'mass': inertia_obj.mass,
                'com': inertia_obj.lever,  # 링크 프레임 원점에서 질량 중심까지의 위치 벡터
                'inertia_tensor': inertia_obj.inertia  # 질량 중심에 대해 표현된 3x3 관성 텐서
            }
            self.link_properties.append(link_data)

            # 2. 조인트의 스크류 축(Screw Axis) 저장
            # joint.motion()은 6D 공간 벡터(Motion vector)를 반환합니다.
            # [wx, wy, wz, vx, vy, vz].T 형태
            joint_obj = self.model.joints[i]
            screw_axis = joint_obj.motion().np  # 6x1 numpy 배열로 변환
            self.screw_axes.append(screw_axis)

        print(f"\nSuccessfully extracted properties for {len(self.link_properties)} moving links.")
        print("Data is stored in 'self.link_properties' and 'self.screw_axes'.")

class KinematicsCalculator:
    def __init__(self, model):
        """
        기구학 계산(자코비안 등)을 위한 클래스 초기화

        Args:
            model: pinocchio로 로드된 로봇 모델 객체
        """
        if model is None:
            raise ValueError("Pinocchio model must be loaded first.")
        self.model = model
        self.data = self.model.createData()

    def compute_jacobian(self, joint_positions, end_effector_frame_name):
        """
        주어진 조인트 위치에 대해 특정 프레임(엔드 이펙터)의 자코비안을 계산합니다.

        Args:
            joint_positions (np.ndarray): 로봇의 현재 조인트 각도(q) 벡터 (DoF 크기)
            end_effector_frame_name (str): 자코비안을 계산할 프레임(링크)의 이름

        Returns:
            np.ndarray: 6xDoF 크기의 자코비안 행렬, 실패 시 None
        """
        if not self.model.existFrame(end_effector_frame_name):
            print(f"Error: Frame '{end_effector_frame_name}' not found in the model.")
            available_frames = [f.name for f in self.model.frames]
            print(f"Available frames are: {available_frames}")
            return None

        # 계산하려는 프레임의 ID를 가져옵니다.
        frame_id = self.model.getFrameId(end_effector_frame_name)
        
        # 자코비안 계산에 앞서 반드시 순기구학이 먼저 계산되어야 합니다.
        # 이 함수는 self.data 내의 모든 프레임 위치/방향 정보를 업데이트합니다.
        pin.forwardKinematics(self.model, self.data, joint_positions)

        # 지정된 프레임의 자코비안을 계산하고 결과를 self.data에 저장합니다.
        # pin.ReferenceFrame.LOCAL_WORLD_ALIGNED는 월드 좌표계와 축이 정렬된, 
        # 그러나 원점은 엔드 이펙터 프레임에 위치한 좌표계에서 자코비안을 표현합니다.
        # 이는 제어에 매우 유용한 형태입니다.
        jacobian = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return jacobian


# --- 실행 및 결과 확인 ---
if __name__ == '__main__':
    # 클래스 인스턴스 생성
    robot_dynamics = DynamicsProperty()
    
    # 정보 추출 메서드 실행
    robot_dynamics.extract_robot_model_info()
    
    # 저장된 데이터 확인
    if robot_dynamics.model:
        print("\n--- Extracted Data Verification ---")
        
        num_links = len(robot_dynamics.link_properties)
        print(f"Total {num_links} link properties stored.")

        # 첫 번째 움직이는 링크(joint 1)의 정보 출력
        if num_links > 0:
            joint_name = robot_dynamics.model.names[1]
            print(f"\nProperties for the link moved by '{joint_name}':")
            
            first_link_props = robot_dynamics.link_properties[0]
            print(f"  - Mass: {first_link_props['mass']:.4f} kg")
            print(f"  - Center of Mass (CoM): {np.round(first_link_props['com'].flatten(), 4)}")
            print(f"  - Inertia Tensor:\n{np.round(first_link_props['inertia_tensor'], 6)}")
            
            first_screw_axis = robot_dynamics.screw_axes[0]
            print(f"  - Screw Axis:\n{np.round(first_screw_axis, 4)}")
