$pip uninstall opencv-contrib-python

$sudo apt-get install libgtk2.0-dev pkg-config

and then type the command below:

$CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

"Allow the building wheel ~20-40 minutes to complete the process"
