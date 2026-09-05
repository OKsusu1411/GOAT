#!/bin/bash

CAN_IF="can0"

# 모터 ID: 1~8  → CAN ID: 0x141 ~ 0x148
START_ID=0x141
END_ID=0x148

echo "=== Set current position to zero ==="
for ((id=START_ID; id<=END_ID; id++)); do
    printf "Motor CAN ID: 0x%X → set zero\n" "$id"
    cansend $CAN_IF $(printf "%03X#1900000000000000" "$id")
    sleep 0.05
done

sleep 0.5

echo "=== Save zero point to ROM ==="
for ((id=START_ID; id<=END_ID; id++)); do
    printf "Motor CAN ID: 0x%X → save to ROM\n" "$id"
    cansend $CAN_IF $(printf "%03X#9100000000000000" "$id")
    sleep 0.05
done

echo "=== Done ==="
