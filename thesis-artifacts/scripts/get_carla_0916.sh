#!/bin/bash
set -eux
TARGET="$HOME/CARLA/standard_0916"
BASE="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux"
mkdir -p "$TARGET/Import"
cd "$TARGET"
wget -c -O CARLA_0916.tar.gz "$BASE/CARLA_0.9.16.tar.gz"
tar -xzf CARLA_0916.tar.gz
cd Import
wget -c -O AdditionalMaps_0.9.16.tar.gz "$BASE/AdditionalMaps_0.9.16.tar.gz"
cd ..
bash ImportAssets.sh
rm -f CARLA_0916.tar.gz
echo "CARLA_SETUP_DONE"
