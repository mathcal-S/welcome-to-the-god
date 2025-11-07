#!/bin/bash
# revive_qiskit.sh - Summon Qiskit in Termux via proot-distro Ubuntu

set -e

echo "🌌 Summoning Qiskit Forge in Termux..."

# Install proot-distro if missing
if ! command -v proot-distro &> /dev/null; then
    pkg update -y
    pkg install proot-distro -y
fi

# Install Ubuntu if missing
if ! proot-distro list | grep -q ubuntu; then
    proot-distro install ubuntu
fi

# Enter Ubuntu and install Qiskit + deps
proot-distro login ubuntu -- bash -c "
apt update -y && apt upgrade -y
apt install python3 python3-pip python3-venv -y
pip3 install --upgrade pip
pip3 install qiskit qiskit-aer qiskit-ibm-runtime flask numpy scikit-optimize dilithium-py
echo 'Qiskit installed in Ubuntu proot. Use proot-distro login ubuntu -- pip3 list to verify.'
"

# Alias for main env (run Qiskit via proot)
echo 'alias qiskit-run="proot-distro login ubuntu -- python3 -c"' >> ~/.bashrc
echo 'alias qiskit-pip="proot-distro login ubuntu -- pip3"' >> ~/.bashrc
source ~/.bashrc

echo "✅ Qiskit summoned! Test: proot-distro login ubuntu -- python3 -c 'import qiskit; print(qiskit.__version__)'"
echo "For Flask: proot-distro login ubuntu -- python3 flask_api.py"
