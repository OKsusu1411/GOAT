#!/usr/bin/env bash

set -e

IFACE="${1:-can0}"
BITRATE="${2:-5000000}"

echo "[*] Setting up CAN interface: ${IFACE} @ ${BITRATE} bps"

# Native Jetson CAN driver
sudo modprobe can
sudo modprobe can_raw
sudo modprobe mttcan

# Native CAN interface must exist
if ! ip link show "${IFACE}" &>/dev/null; then
    echo "[FAIL] ${IFACE} does not exist."
    echo "Check mttcan driver / device tree / pinmux."
    exit 1
fi

echo "[*] Bringing ${IFACE} down..."
sudo ip link set "${IFACE}" down || true

echo "[*] Configuring ${IFACE}..."
sudo ip link set "${IFACE}" type can \
    bitrate "${BITRATE}" \
    restart-ms 100 \
    berr-reporting on

echo "[*] Bringing ${IFACE} up..."
sudo ip link set "${IFACE}" up

sleep 0.5

echo
echo "=== ip -details -statistics link show ${IFACE} ==="
ip -details -statistics link show "${IFACE}"

echo

if ip link show "${IFACE}" | grep -q "UP"; then
    echo "[OK] ${IFACE} is UP and configured at ${BITRATE} bps."
else
    echo "[FAIL] ${IFACE} is NOT UP."
    exit 1
fi

if command -v candump &>/dev/null; then
    echo
    echo "Tip: candump ${IFACE}"
fi
