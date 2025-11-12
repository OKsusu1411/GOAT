#ifndef VYB_CONTROLLER_H
#define VYB_CONTROLLER_H

#include <Arduino.h>
#include <ArduinoEigenDense.h>
#include "Params.h"
#include "MGServo.h"

/**
 * @class VYBController
 * @brief 제어 시스템의 LQR 기반 동작을 처리하는 클래스
 */
class VYBController {
private:
  MGServo& ServoRW;  ///< 우측 휠 서보
  MGServo& ServoLW;  ///< 좌측 휠 서보

  std::vector<Eigen::Matrix<float, 2, 4>> Ks;  ///< LQR 게인 행렬들의 벡터
  Eigen::Matrix<float, 2, 4> K;                ///< 현재 사용 중인 LQR 게인
  Eigen::Matrix<float, 2, 1> u;                ///< 제어 입력 벡터

  float iq_factor;        ///< 전류 변환 계수 (A/LSB)
  float torque_constant;  ///< 토크 상수 (Nm/A)
  float LW_factor = 0.001043224f;
  float RW_factor = 0.000857902f;

  float AngleFixRate = 0.1;

  float saturation;  ///< input saturation
  Eigen::Matrix<float, 2, 1> saturation_vec;
  const int RW_bias = 12;  ///< 우측 휠 모터의 바이어스 값

public:
  float theta_d;
  /**
   * @brief 생성자: VYBController 초기화
   * @param ServoRW_ 우측 휠 서보 객체 참조
   * @param ServoLW_ 좌측 휠 서보 객체 참조
   */
  VYBController(MGServo& ServoRW_, MGServo& ServoLW_)
    : ServoRW(ServoRW_), ServoLW(ServoLW_) {
    theta_d = 0.f;
    // 전류 및 토크 상수 초기화
    iq_factor = 0.001611328f;  // (A/LSB) 3.3 / 2048
    torque_constant = 0.7f;    // (Nm/A) * reduction ratio
    saturation = iq_factor * torque_constant * MAX_TORQUE_COMMAND;
    // saturation_vec << MAX_TORQUE_COMMAND * RW_factor, MAX_TORQUE_COMMAND * LW_factor;
    saturation_vec << MAX_TORQUE, MAX_TORQUE;


    // LQR 게인 초기화 (하드코딩된 데이터 삽입)
    Eigen::Matrix<float, 2, 4> mat;
    //////////////////////////////////////////////////////////
    mat << 1.35316904f, 0.14009245f, 0.23358589f, -0.12161902f,
      -1.36517013f, -0.14243867f, -0.23405192f, -0.12179830f;
    Ks.push_back(mat);

    mat << 1.40047216f, 0.14638508f, 0.23369175f, -0.12160757f,
      -1.41285639f, -0.14885450f, -0.23407320f, -0.12184087f;
    Ks.push_back(mat);

    mat << 1.44602994f, 0.15279748f, 0.23382116f, -0.12158976f,
      -1.45881851f, -0.15539539f, -0.23413544f, -0.12186834f;
    Ks.push_back(mat);

    mat << 1.48968650f, 0.15931475f, 0.23397350f, -0.12156787f,
      -1.50289679f, -0.16204545f, -0.23423617f, -0.12188368f;
    Ks.push_back(mat);

    mat << 1.53149557f, 0.16592279f, 0.23414424f, -0.12154222f,
      -1.54514525f, -0.16879039f, -0.23436934f, -0.12188799f;
    Ks.push_back(mat);

    mat << 1.57156576f, 0.17260877f, 0.23432853f, -0.12151281f,
      -1.58567342f, -0.17561742f, -0.23452873f, -0.12188200f;
    Ks.push_back(mat);

    mat << 1.61001973f, 0.17936153f, 0.23452203f, -0.12147964f,
      -1.62460463f, -0.18251548f, -0.23470875f, -0.12186644f;
    Ks.push_back(mat);

    mat << 1.64697959f, 0.18617179f, 0.23472118f, -0.12144289f,
      -1.66206114f, -0.18947534f, -0.23490463f, -0.12184212f;
    Ks.push_back(mat);

    mat << 1.68256117f, 0.19303219f, 0.23492316f, -0.12140290f,
      -1.69815838f, -0.19648966f, -0.23511244f, -0.12180996f;
    Ks.push_back(mat);

    mat << 1.71687199f, 0.19993741f, 0.23512588f, -0.12136017f,
      -1.73300286f, -0.20355303f, -0.23532897f, -0.12177102f;
    Ks.push_back(mat);

    mat << 1.75001117f, 0.20688435f, 0.23532789f, -0.12131532f,
      -1.76669221f, -0.21066222f, -0.23555173f, -0.12172640f;
    Ks.push_back(mat);

    mat << 1.78207150f, 0.21387275f, 0.23552830f, -0.12126915f,
      -1.79931713f, -0.21781670f, -0.23577883f, -0.12167735f;
    Ks.push_back(mat);

    mat << 1.81314599f, 0.22090653f, 0.23572687f, -0.12122289f,
      -1.83096725f, -0.22501991f, -0.23600886f, -0.12162555f;
    Ks.push_back(mat);

    mat << 1.84334794f, 0.22799757f, 0.23592427f, -0.12117958f,
      -1.86174765f, -0.23228239f, -0.23624048f, -0.12157472f;
    Ks.push_back(mat);
    /////////////////////////////////////////////////////////////////
  }

  /**
   * @brief 현재 제어 입력 벡터를 반환
   * @return Eigen::Matrix<float, 2, 1> u
   */
  Eigen::Matrix<float, 2, 1> getInputVector() {
    return u;
  }

  /**
   * @brief 모터 속도 측정을 수행하여 측정 벡터에 반영
   * @param z 모터 속도 측정값을 저장할 벡터
   */
  void getMotorSpeedMeasurement(Eigen::Matrix<float, 8, 1>& z) {
    z(6) = ServoRW.getMotorSpeed() * M_PI / 180;
    z(7) = ServoLW.getMotorSpeed() * M_PI / 180;
  }

  /**
  * @brief 모터 Current 측정값 update
  * @param iq_vec 모터 current 측정값을 저장할 벡터
  */
  void getMotorCurrentMeasurement(Eigen::Matrix<float, 2, 1>& iq_vec) {
    iq_vec << ServoRW.getMotorIq(), ServoLW.getMotorIq();
  }

  /**
   * @brief 현재 높이에 따라 LQR 게인 K를 계산
   * @param h 현재 높이 (m)
   */
  void computeGainK(const float h) {
    float temp = (h - HEIGHT_MIN) / 0.01;  // 구간을 10mm당 하나씩 나눔
    int idx = static_cast<int>(temp);      // 구간의 정수 인덱스 계산

    if (idx >= 0 && idx < static_cast<int>(Ks.size()) - 1) {
      // 보간 비율 계산
      float ratio = temp - idx;  // 현재 위치가 구간 내에서 차지하는 비율

      // 보간 수행
      K = Ks.at(idx) * (1.0f - ratio) + Ks.at(idx + 1) * ratio;
    } else if (idx < 0) {
      // h가 HEIGHT_MIN 이하일 경우 최소값 사용
      K = Ks.front();
    } else {
      // h가 범위를 벗어날 경우 최대값 사용
      K = Ks.back();
    }
  }

  /**
   * @brief 상태 벡터를 기반으로 제어 입력 벡터를 계산
   * @param x_d 목표 상태 벡터
   * @param x 현재 상태 벡터
   */
  void computeInput(Eigen::Matrix<float, 4, 1>& x_d, Eigen::Matrix<float, 4, 1>& x) {

    // if (x_d(0) - x(0) < 0) {
    //   theta_d -= AngleFixRate * dt;
    // } else {
    //   theta_d += AngleFixRate * dt;
    // }
    // x_d(0) = theta_d;

    u = K * (x_d - x);

    // // Input saturation
    // for (int j = 0; j < 2; j++) {
    //   if (u(j) > saturation) {
    //     u(j) = saturation;
    //   } else if (u(j) < -saturation) {
    //     u(j) = -saturation;
    //   }
    // }
    // Input saturation
    for (int j = 0; j < 2; j++) {
      if (u(j) > saturation_vec(j)) {
        u(j) = saturation_vec(j);
      } else if (u(j) < -saturation_vec(j)) {
        u(j) = -saturation_vec(j);
      }
    }
  }

  /**
   * @brief 계산된 제어 명령을 서보에 전송
   */
  void sendControlCommand() {
    // float u_RW = u(0) / (iq_factor * torque_constant);
    // float u_LW = u(1) / (iq_factor * torque_constant);

    // // Right Wheel motor의 마찰로 인해 발생하는 torque 문제를 조정해줌
    // if (u_RW < 0) {
    //   u_RW -= RW_bias;
    // } else if (u_RW > 0) {
    //   u_RW += RW_bias;
    // }

    float u_RW = u(0) / RW_factor;
    float u_LW = u(1) / LW_factor;

    ServoRW.sendTorqueControlCommand(static_cast<int16_t>(u_RW));
    ServoLW.sendTorqueControlCommand(static_cast<int16_t>(u_LW));
  }

  void sendDirectControlCommand(Eigen::Matrix<float, 2, 1> u_) {
    float u_RW = u_(0) / (iq_factor * torque_constant);
    float u_LW = u_(1) / (iq_factor * torque_constant);

    // Right Wheel motor의 마찰로 인해 발생하는 torque 문제를 조정해줌
    if (u_RW < 0) {
      u_RW -= RW_bias;
    } else if (u_RW > 0) {
      u_RW += RW_bias;
    }

    ServoRW.sendTorqueControlCommand(static_cast<int16_t>(u_RW));
    ServoLW.sendTorqueControlCommand(static_cast<int16_t>(u_LW));
  }

  void sendReadStateCommand() {
    ServoRW.sendCommandReadMotorState2();
    ServoLW.sendCommandReadMotorState2();
  }
};

#endif  // VYB_CONTROLLER_H