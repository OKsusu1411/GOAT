[Reference blog] : (https://blog.naver.com/chandong83/223407505153)


# CAN0 현재 상태 확인
```bash
sudo ip -detail link show can0
```

# 설정을 위해 can0 정지
```bash
sudo ip link set can0 down
```
# 100Kbps로 설정
``bash`
sudo ip link set can0 type can bitrate 100000
```
# bitrate가 바뀌었는지 확인
```bash
sudo ip -detail link show can0
```
# can 시작
```bash
sudo ip link set can0 up
```
# can scan 시작
```bash
candump can0 &
```
# 데이터 전송
```bash
sudo cansend can1 123#1122334455667788 
```
