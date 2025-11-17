#!/usr/bin/env bash

# Usage:
#   ./setup_can.sh                 # uses can0, 1Mbps by default
#   ./setup_can.sh can1 500000     # uses can1, 500 kbps

set -e

IFACE=${1:-can0}          # CAN interface name (default: can0)
BITRATE=${2:-1000000}     # Bitrate in bps (default: 1 Mbps)

echo "[*] Setting up CAN interface: ${IFACE} @ ${BITRATE} bps"

# If the interface already exists, bring it down first
if ip link show "${IFACE}" &>/dev/null; then
    echo "[*] Bringing ${IFACE} down (if it is up)..."
    sudo ip link set "${IFACE}" down || true
fi

# Configure the interface as CAN with the given bitrate
echo "[*] Configuring ${IFACE}..."
sudo ip link set "${IFACE}" type can bitrate "${BITRATE}" restart-ms 100

# Bring the interface up
echo "[*] Bringing ${IFACE} up..."
sudo ip link set "${IFACE}" up

sleep 0.5

echo
echo "=== ip -details link show ${IFACE} ==="
ip -details link show "${IFACE}"

echo
# Check if the interface state is UP
if ip -details link show "${IFACE}" | grep -q "state UP"; then
    echo "[OK] ${IFACE} is UP and configured at ${BITRATE} bps."
    EXIT_CODE=0
else
    echo "[FAIL] ${IFACE} is NOT UP. Please check wiring / device / bitrate."
    EXIT_CODE=1
fi

# If can-utils is installed, print a small tip
if command -v candump &>/dev/null; then
    echo
    echo "Tip: you can monitor CAN traffic with:"
    echo "  candump ${IFACE}"
fi

exit ${EXIT_CODE}
