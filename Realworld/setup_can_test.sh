#!/usr/bin/env bash

# Usage:
#   ./setup_can.sh             # uses can0 and can1, 1Mbps by default
#   ./setup_can.sh 500000      # uses can0 and can1, 500 kbps

set -e

INTERFACES=("can0" "can1")
BITRATE=${1:-1000000}     
TXQUEUELEN=1000

echo "=================================================="
echo "[*] Setting up CAN interfaces: ${INTERFACES[*]}"
echo "[*] Target Bitrate: ${BITRATE} bps"
echo "=================================================="

EXIT_CODE=0

# Setting each CAN bus
for IFACE in "${INTERFACES[@]}"; do
    echo "[*] Processing ${IFACE}..."

    # If the interface already exists, bring it down first
    if ip link show "${IFACE}" &>/dev/null; then
        echo "    - Bringing ${IFACE} down (if it is up)..."
        sudo ip link set "${IFACE}" down || true
    fi

    # Configure the interface as CAN with the given bitrate
    echo "    - Configuring ${IFACE}..."
    sudo ip link set "${IFACE}" type can bitrate "${BITRATE}" restart-ms 100

    # Bring the interface up
    echo "    - Bringing ${IFACE} up with txqueuelen ${TXQUEUELEN}..."
    sudo ip link set "${IFACE}" up txqueuelen "${TXQUEUELEN}"

    sleep 0.2

    # Check if the interface state is UP
    if ip -details link show "${IFACE}" | grep -q "state UP"; then
        echo "    -> [OK] ${IFACE} is UP and ready."
    else
        echo "    -> [FAIL] ${IFACE} is NOT UP. Check wiring or device."
        EXIT_CODE=1
    fi
    echo "--------------------------------------------------"
done

# If can-utils is installed, print a small tip
if command -v candump &>/dev/null; then
    echo
    echo "Tip: you can monitor CAN traffic with:"
    echo "  candump can0"
    echo "  candump can1"
fi

exit ${EXIT_CODE}