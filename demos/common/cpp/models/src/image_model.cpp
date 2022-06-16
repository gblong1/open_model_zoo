/*
// Copyright (C) 2021-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/image_model.h"

#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize, const std::string& layout)
    : ModelBase(modelFileName, layout),
      useAutoResize(useAutoResize) {}

template <class T>
static inline void CopyToTensor(const std::uint8_t* pSourceImg, T* pTensor,
    const size_t width, const size_t height, const size_t channels)
{
    for (int p = 0; p < width * height * channels; p++)
    {
        pTensor[p] = (T)pSourceImg[p];
    }
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    auto img = inputTransform(origImg);

    if (!useAutoResize) {
        // /* Resize and copy data from the image to the input tensor */
        //const ov::Tensor& frameTensor = request.get_tensor(inputsNames[0]);  // first input should be image
        const ov::Tensor& frameTensor = request.get_input_tensor();  // first input should be image
        const ov::Shape& tensorShape = frameTensor.get_shape();
        //const ov::Layout layout("NHWC");
        const ov::Layout layout = getLayoutFromShape(tensorShape);
        
        const size_t width = tensorShape[ov::layout::width_idx(layout)];
        const size_t height = tensorShape[ov::layout::height_idx(layout)];
        const size_t channels = tensorShape[ov::layout::channels_idx(layout)];

        //slog::info << "Shape" << tensorShape <<  ", DIMS = " << "1," << channels << "," << height << "," << width << slog::endl;

        if (static_cast<size_t>(img.channels()) != channels) {
            throw std::runtime_error("The number of channels for model input and image must match");
        }
        if (channels != 1 && channels != 3) {
            throw std::runtime_error("Unsupported number of channels");
        }
        img = resizeImageExt(img, width, height);
    }

    //For some reason when using VPU pre-compiled models, this 'wrapMat2Tensor'
    // sequence will produce tensors with incompatible layout and/or strides,
    // which generates bogus results. TODO: Understand why..
#if 0
    ov::Tensor wrapped_tensor = wrapMat2Tensor(img);
    request.set_tensor(inputsNames[0], wrapped_tensor);
#else
    // Instead, extract the resized image from the cv::Mat, 'img', and copy it
    // to the input tensor.
    {
        ov::Tensor input_tensor = request.get_input_tensor();

        const ov::Shape& tensorShape = input_tensor.get_shape();
        const ov::Layout layout = getLayoutFromShape(tensorShape);
        const size_t width = tensorShape[ov::layout::width_idx(layout)];
        const size_t height = tensorShape[ov::layout::height_idx(layout)];
        const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
        ov::element::Type elemType = input_tensor.get_element_type();

        std::uint8_t* pInputImg = (std::uint8_t *)img.ptr();

#define TENSOR_COPY_CASE(elem_type)                                                       \
        case ov::element::Type_t::elem_type: {                                            \
            using tensor_type = ov::fundamental_type_for<ov::element::Type_t::elem_type>; \
            tensor_type *pTensor = input_tensor.data<tensor_type>();                      \
            CopyToTensor<tensor_type>(pInputImg, pTensor, width, height, channels);       \
            break;                                                                        \
        }

        switch (elemType)
        {
            TENSOR_COPY_CASE(f32);
            TENSOR_COPY_CASE(f64);
            TENSOR_COPY_CASE(f16);
            TENSOR_COPY_CASE(i16);
            TENSOR_COPY_CASE(u8);
            TENSOR_COPY_CASE(i8);
            TENSOR_COPY_CASE(u16);
            TENSOR_COPY_CASE(i32);
            TENSOR_COPY_CASE(u32);
            TENSOR_COPY_CASE(i64);
            TENSOR_COPY_CASE(u64);
        default:
            OPENVINO_ASSERT(false, "Unsupported input tensor type for image copy ", elemType);
        }
    }
#endif

    return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}
