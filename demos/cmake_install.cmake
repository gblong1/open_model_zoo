# Install script for directory: D:/open_model_zoo/demos

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/Demos")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/open_model_zoo/demos/thirdparty/gflags/cmake_install.cmake")
  include("D:/open_model_zoo/demos/common/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/common/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/multi_channel_common/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/background_removal_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/background_subtraction_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/classification_benchmark_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/crossroad_camera_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/face_detection_mtcnn_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/gaze_estimation_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/gaze_estimation_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/gesture_recognition_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/human_pose_estimation_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/image_processing_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/interactive_face_detection_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/interactive_face_detection_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/mask_rcnn_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/mri_reconstruction_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/multi_channel_face_detection_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/multi_channel_human_pose_estimation_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/multi_channel_object_detection_demo_yolov3/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/noise_suppression_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/object_detection_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/pedestrian_tracker_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/security_barrier_camera_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/segmentation_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/smart_classroom_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/smart_classroom_demo/cpp_gapi/cmake_install.cmake")
  include("D:/open_model_zoo/demos/social_distance_demo/cpp/cmake_install.cmake")
  include("D:/open_model_zoo/demos/text_detection_demo/cpp/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/open_model_zoo/demos/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
