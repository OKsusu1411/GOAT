#ifndef CAN_H_
#define CAN_H_

#include <iostream>
#include <thread>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

class Can {
  public:
  // CAN bus open & close
  void open();
  void close();

  

};
#endif // CAN_H_