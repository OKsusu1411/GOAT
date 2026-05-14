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

# Setting each CAN
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

    # Wait up to ~1s for the netdev IFF_UP flag to appear.
    # We check the flag list "<...,UP,...>" on the first line of `ip link show`,
    # NOT the "state UP" token: for SocketCAN that token reflects the CAN
    # controller state (ERROR-ACTIVE / BUS-OFF / ...) and may not read "UP"
    # even when the interface is administratively up and the transceiver is on.
    UP_OK=0
    for _ in 1 2 3 4 5; do
        # Match IFF_UP, anchored so it can't hit the "UP" inside "LOWER_UP".
        if ip link show "${IFACE}" 2>/dev/null | head -n1 | grep -qE '[<,]UP[,>]'; then
            UP_OK=1
            break
        fi
        sleep 0.2
    done

    if [ "${UP_OK}" -eq 1 ]; then
        # CAN controller state, logged for visibility only; not part of the UP/DOWN decision.
        CAN_STATE=$(ip -details link show "${IFACE}" | awk '/can state/ {print $3; exit}')
        echo "    -> [OK] ${IFACE} is UP and ready (can state: ${CAN_STATE:-unknown})."
    else
        echo "    -> [FAIL] ${IFACE} is NOT UP. Check wiring, termination, or device."
        # Dump details to help diagnose (BUS-OFF / STOPPED / ERROR-PASSIVE differ a lot).
        ip -details link show "${IFACE}" || true
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