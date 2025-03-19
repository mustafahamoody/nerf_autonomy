#!/bin/bash

# Source ROS and workspace
source "/opt/ros/humble/setup.bash"
source "${WORKSPACE}/install/setup.bash"

# Add bash configuration to the user's .bashrc during container startup
if [ -z "$BASH_CONFIG_ADDED" ]; then
  cat >> ~/.bashrc << 'EOF'
# Function to run ROS2 commands with proper error handling
run_ros2_command() {
  set -o pipefail
  "$@" 2>&1 | tee /tmp/last_command.log
  exit_code=${PIPESTATUS[0]}
  if [ $exit_code -ne 0 ]; then
    echo -e "\n\033[31mCommand failed with exit code $exit_code\033[0m"
    echo "Error output captured in /tmp/last_command.log"
    return $exit_code
  fi
  return 0
}

# Set up better bash error handling
trap 'echo "Command failed with status $?: $BASH_COMMAND"; true' ERR

# Create aliases for common ROS2 commands
alias ros2run='run_ros2_command ros2 run'
alias ros2launch='run_ros2_command ros2 launch'

# Add a welcome message to show the shell is ready
echo -e "\033[32mROS2 development environment ready\033[0m"
EOF
  export BASH_CONFIG_ADDED=1
fi

# Execute the passed command
exec "$@"