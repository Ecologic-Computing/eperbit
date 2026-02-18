#!/bin/bash
sudo timedatectl set-ntp true
timedatectl set-timezone Europe/Berlin
date -s "$(curl -sI https://google.com | grep -i '^date:' | cut -d' ' -f2-)"

