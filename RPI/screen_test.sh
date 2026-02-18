#!/usr/bin/env bash

sudo sleep 10
echo "sleeping for 10s and launching test in screen session detached background"
sudo screen -S test bash -c  "./test.sh"
