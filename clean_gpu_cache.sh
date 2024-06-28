#!/bin/bash

# Find all Python processes
processes=$(ps aux | grep python | grep -v grep | awk '{print $2}')

# Check if there are any processes to kill
if [ -z "$processes" ]; then
  echo "No Python processes found."
else
  # Kill each process
  for pid in $processes
  do
    echo "Killing process ID: $pid"
    kill -9 $pid
  done
fi