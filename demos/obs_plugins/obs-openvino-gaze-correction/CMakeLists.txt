# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

message("obs background removal")

file(GLOB_RECURSE SRC_FILES ./*.cpp)
file(GLOB_RECURSE H_FILES ./*.h)
file(GLOB_RECURSE HPP_FILES ./*.hpp)

add_obs_plugin(NAME obs-gaze-correction-ov
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES} ${HPP_FILES}
	INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DEPENDENCIES monitors utils)
