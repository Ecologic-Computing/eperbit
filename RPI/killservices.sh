#!/usr/bin/env bash

set -e
# Set time properly
sudo /home/rpi/time.sh

sudo rfkill block wifi; sudo rfkill block bluetooth
# Enable performance mode on RPI5 to run at 2.4 GHz always
#echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set Performance Mode
sudo bash -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$c"; done'
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor


LOGDIR="$HOME/pi-benchmark-logs-$(date +%F-%H%M%S)"
mkdir -p "$LOGDIR"

echo "Logging to $LOGDIR"

#####################################
# 1. Log system info
#####################################
{
  echo "### OS INFO"
  uname -a
  cat /etc/os-release
  echo
  echo "### CPU INFO"
  lscpu
  echo
  echo "### MEMORY"
  free -h
  echo
  echo "### DISK"
  df -h
} > "$LOGDIR/system-info.txt"

#####################################
# 2. Log running services
#####################################
systemctl list-units --type=service --state=running \
  > "$LOGDIR/running-services.txt"

#####################################
# 3. Log running processes
#####################################
ps auxf > "$LOGDIR/processes.txt"

#####################################
# 4. Define safe-to-stop services
#####################################
SAFE_SERVICES=(
  accounts-daemon
  avahi-daemon
  bluetooth
  cups
  cups-browsed
  ModemManager
  triggerhappy
  udisks2
  rtkit-daemon
  cron
  getty@tty1
  serial-getty@ttyAMA10
  avahi-daemon
  bluetooth
  accounts-daemon
  bluetoothd
  hciuart
  ModemManager
  triggerhappy
  cups
  cups-browsed
  alsa-state
  pipewire
  pipewire-pulse
  pulseaudio
  lightdm
  gdm
  sddm
  plymouth
  apt-daily
  apt-daily-upgrade
)

#####################################
# 5. Stop services if running
#####################################
echo "Stopping optional services..." | tee "$LOGDIR/actions.log"

for svc in "${SAFE_SERVICES[@]}"; do
  if systemctl is-active --quiet "$svc"; then
    echo "Stopping $svc" | tee -a "$LOGDIR/actions.log"
    sudo systemctl stop "$svc"
  else
    echo "$svc not running" >> "$LOGDIR/actions.log"
  fi
done

#####################################
# 6. Verify SSH is alive
#####################################
if systemctl is-active --quiet ssh; then
  echo "SSH is running ✔" | tee -a "$LOGDIR/actions.log"
else
  echo "WARNING: SSH NOT RUNNING!" | tee -a "$LOGDIR/actions.log"
fi

#####################################
# 7. Final snapshot
#####################################
systemctl list-units --type=service --state=running \
  > "$LOGDIR/running-services-after.txt"

echo "Benchmark prep complete."
echo "Logs saved in $LOGDIR"
