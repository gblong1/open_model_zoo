# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

message("obs background removal")

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)

add_obs_plugin(NAME obs_background_removal_ov
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES}
    DEPENDENCIES monitors models pipelines)
