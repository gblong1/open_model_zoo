/*
// Copyright (C) 2018-2021 Intel Corporation
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
#include <math.h>
#include "models/segmentation_model.h"
#include "utils/ocv_common.hpp"

using namespace InferenceEngine;

SegmentationModel::SegmentationModel(const std::string& modelFileName, bool useAutoResize, InferenceEngine::Precision ip) :
    input_precision(ip), ImageModel(modelFileName, useAutoResize) {}

void SegmentationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork)
{
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");

    inputsNames.push_back(inputShapes.begin()->first);

    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(input_precision);

    if (useAutoResize) {
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.setLayout(Layout::NHWC);
    } else {
        inputInfo.setLayout(Layout::NCHW);
    }

#if 0
    auto preProcess = inputInfo.getPreProcess();
    preProcess.init(3);
    preProcess[0]->meanValue = (0.485f);
    preProcess[1]->meanValue = (0.456f);
    preProcess[2]->meanValue = (0.406f);
    preProcess[0]->stdScale = (0.229f);
    preProcess[1]->stdScale = (0.224f);
    preProcess[2]->stdScale = (0.225f);
    preProcess.setVariant(InferenceEngine::MEAN_VALUE);
#endif

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputsDataMap.begin()->first);
    Data& data = *outputsDataMap.begin()->second;

    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    switch (outSizeVector.size()) {
    case 3:
        outChannels = 1;
        outHeight = (int)(outSizeVector[1]);
        outWidth = (int)(outSizeVector[2]);
        break;
    case 4:
        outChannels = (int)(outSizeVector[1]);
        outHeight = (int)(outSizeVector[2]);
        outWidth = (int)(outSizeVector[3]);
        break;
    default:
        throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are supported.");
    }
}


void matU8ToNormalized_0_1_FloatBlob(const cv::Mat& orig_image, const InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();

    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];

    if (static_cast<size_t>(orig_image.channels()) != channels) {
        throw std::runtime_error("The number of channels for net input and image must match");
    }
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
        static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = ((float)resized_image.at<uchar>(h, w))/255.0f;
            }
        }
    }
    else if (channels == 3) {  
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {

                    blob_data[batchOffset + c * width * height + h * width + w] =
                        ((float)resized_image.at<cv::Vec3b>(h, w)[c]) / 255.0f;
                }
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

std::shared_ptr<InternalModelData> SegmentationModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request)
{
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    std::shared_ptr<InternalModelData> resPtr = nullptr;

    if (useAutoResize)
    {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
        /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
        resPtr = std::make_shared<InternalImageMatModelData>(img);
    }
    else
    {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        if (input_precision == InferenceEngine::Precision::U8)
        {
            matU8ToBlob<uint8_t>(img, frameBlob);
        }
        else if (input_precision == InferenceEngine::Precision::FP32)
        {
            matU8ToNormalized_0_1_FloatBlob(img, frameBlob);
        }

        resPtr = std::make_shared<InternalImageModelData>(img.cols, img.rows);
    }

    return resPtr;
}

std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult(infResult.frameId, infResult.metaData);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();

    MemoryBlob::Ptr blobPtr = infResult.getFirstOutputBlob();

    void* pData = blobPtr->rmap().as<void*>();

    result->resultImage = cv::Mat(outHeight, outWidth, CV_8UC1);

    if (outChannels == 1 && blobPtr->getTensorDesc().getPrecision() == Precision::I32)
    {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1, pData);
        predictions.convertTo(result->resultImage, CV_8UC1);
    }
    else if (outChannels == 1 && blobPtr->getTensorDesc().getPrecision() == Precision::FP32)
    {
        //for (int i = 0; i < 20; i++)
        //{
        //    std::cout << "pData[" << i << "] = " << ((float*)pData)[i] << std::endl;
        //}

        cv::Mat fg_confidence(outHeight, outWidth, CV_32FC1, pData);

        //for (int i = 0; i < 20; i++)
       // {
       //     std::cout << "fg_confidence[" << i << "] = " << ((float*)fg_confidence.ptr(0))[i] << std::endl;
        //}

        fg_confidence.convertTo(result->resultImage, CV_8UC1, 255.0);

        //for (int i = 0; i < 20; i++)
        //{
       //     std::cout << "result before thresh[" << i << "] = " << (int)((uchar*)result->resultImage.ptr(0))[i] << std::endl;
       // }

         result->resultImage = (result->resultImage > 128) & 15;

       // for (int i = 0; i < 20; i++)
       // {
       //     std::cout << "result[" << i << "] = " << (int)((uchar*)result->resultImage.ptr(0))[i] << std::endl;
       // }

    }
    else if (blobPtr->getTensorDesc().getPrecision() == Precision::FP32)
    {
        float* ptr = reinterpret_cast<float*>(pData);
        for (int rowId = 0; rowId < outHeight; ++rowId)
        {
            for (int colId = 0; colId < outWidth; ++colId)
            {
                int classId = 0;
                float maxProb = -1.0f;
                for (int chId = 0; chId < outChannels; ++chId)
                {
                    float prob = ptr[chId * outHeight * outWidth + rowId * outWidth + colId];
                    if (prob > maxProb)
                    {
                        classId = chId;
                        maxProb = prob;
                    }
                } // nChannels

                result->resultImage.at<uint8_t>(rowId, colId) = classId;
            } // width
        } // height
    }

    cv::resize(result->resultImage, result->resultImage,
        cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),
        0, 0, cv::INTER_NEAREST);

    return std::unique_ptr<ResultBase>(result);
}
