# Reinforcement Learning Sim2Real Transfer Manual
Reinforcement Learning으로 학습된 policy를 실제 robot에 deployment하는 피와 땀으로 쓰여진 Manual

## Checklist
Sim2real transfer를 하기 위한 필수 checklist  
- [ ] **Sim, real action, observation space 완벽 일치하는지 확인**
- [ ] **All sensor's unit sim, real 일치**
- [ ] **Sensor initial data(wheel position, ...) sim, real 통일**
- [ ] **Sensor calibration 진행(real 에서)** 
- [ ] **Motor revolution direction sim, real 일치**
- [ ] **Motor gear ratio joint state, torque에 적용(real에서)**
- [ ] **Motor torque constant가 잘 추종하는지 확인(real에서)**
- [ ] **IMU sensor axis sim, real 일치**
- [ ] **Policy와 low-level controller loop synchronization(real에서)**
- [ ] **Actor network output이 sim, real동일한지 확인** 
- [ ] **Action scaling, observation standardization sim, real 동일한지 확인**
- [ ] **Torque limiter, joint velocity limiter, joint position limiter 등의 safety limiter 완비(real 에서)**
- [ ] **Natural standing configuration을 nominal controller가 잘 유지하는지 확인(real에서)**
- [ ] **Timestep을 observation으로 두는 task의 경우 real에서 timer 초기화 확인**
- [ ] **Sensor, Actuator write, read frequency 가 simulation과 같은지 확인**

## Tips
- Gait robot의 경우 natural하게 서있는 **Natural Standing Configuration(NSC)** 을 미리 설정한 후 sim에서 initial pose로써 학습한다.
- Nominal controller로 NSC를 유지하며 서있다가 policy를 실행하여 task를 진행하는 방식을 많이 사용한다.
- Real에서 NSC를 간신히(compliance하게) 유지할 수 있는 low-level controller gain을 실험적으로 구한 후 이를 실제 gain으로 사용
- Domain Randomization(DR)과 modeling을 잘 했다면 위에서 구한 controller gain을 sim에 그대로 사용하여 학습을 진행해도 된다고 한다.
- Action space와 low-level controller는 policy가 robot을 제어하는 interface로써 작용하는 역할, 즉 sim에서만 학습 후 deployment하는 경우 크게 중요한 파트가 아니다.
- Task space action은 IK solver의 존재에 의해 별로 선호되지 않는다. 특히 gait robot은 더욱 그러하다.
- Joint space action의 경우 대게 **Reference controller input = NSC's joint position + Action** 형태의 delta position action을 사용한다.
- Joint position, velocity limit violation의 경우 limit margin을 설정하고 이를 위반했을 시 그냥 꺼버리는 등의 강한 제재를 가한다. 이런 safety limiter는 굳이 sim에도 적용할 필요는 없다고 한다. 애초에 limit을 위반하는 경우는 학습 완료 후 거의 없어야 함.
- Linear acceleration은 observation으로 잘 두지 않는다. 그 이유는 impacts, noise등의 여러 disturbances가 많기 때문
- DR은 robot's property(link mass, link length, joint friction, ...) 등 정말 다양하게 가하여 sim에서 잘 작동하는 policy를 뽑은 후 DR을 하나씩 줄여가며 policy를 최적화한다.
- NSC가 높게 서있을 수록 제어가 어렵고, 낮을 수록 actuator에 무리가 간다. 로봇의 spec을 고려하여 선정해야함.
- Curriculum level은 성공률을 통해 제어하기도 하고, 경험적으로 iteration 수에 따라 증가시키기도 한다.
- 자잘한 observation data, sensor unit 불일치가 굉장히 많을 것이므로 **무조건 이에 대한 전수조사를 꼼꼼히 진행해야한다.** 
- Motor gear ratio 적용은 motor io 관리하는 program인 최하단에서 관리하는게 편함. 그래서 모든 Control logic에선 Joint 기준으로 계산 후 최하단에서만 gear비로 motor 기준으로 변환.    
- 2족 보행 Robot의 경우 Nominal controller를 만들기 쉽지않다. Inverse Dynamics + LIPM controller를 사용했으나 overshoot 등의 문제로 인해 Policy로 구현하는 것이 낫다고 판단.   
- 2족 보행 Robot의 일관된 Initial pose구현을 위해 stand를 제작하여 올려두는 게 유리하다. 왜냐하면 NSC를 위한 Nominal controller, Task policy 모두 intial pose가 일관되어야 작동이 원활하기 때문이다. (Nominal controller 상단의 trajectory generator도 따로 만들어야하기 때문)  


