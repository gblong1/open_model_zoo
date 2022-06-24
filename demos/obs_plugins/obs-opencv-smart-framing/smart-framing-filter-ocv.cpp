#include <obs-module.h>
#include <media-io/video-scaler.h>

#ifdef _WIN32
#include <wchar.h>
#endif

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils_gapi/stream_source.hpp>
#include <utils/args_helper.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/default_flags.hpp>
#include <models/detection_model.h>

#include "custom_kernels.hpp"

#include "plugin-macros.generated.h"


//const char* MODEL_SINET = "SINet_Softmax_simple.onnx";
//const char* MODEL_MODNET = "modnet_simple.onnx";
//const char* MODEL_MEDIAPIPE = "mediapipe.onnx";
//const char* MODEL_SELFIE = "selfie_segmentation.onnx";
//const char* MODEL_RVM = "rvm_mobilenetv3_fp32.onnx";
const char* MODEL_DEEPLABV3 = "deeplabv3.xml";

const char* DEVICE_CPU = "CPU";
const char* DEVICE_VPU = "VPUX";
const char* DEVICE_GPU = "GPU";

std::string modelFilepathS = "D:\\clientmodels\\BDK4\\yolo-v4-tiny\\tf\\FP16-INT8\\yolo-v4-tiny.xml";
//AsyncPipeline pipeline;

namespace util {

    static cv::gapi::GKernelPackage getKernelPackage(const std::string& type) {
        if (type == "opencv") {
            return cv::gapi::combine(cv::gapi::core::cpu::kernels(),
                cv::gapi::imgproc::cpu::kernels());
        }
        else if (type == "fluid") {
            return cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                cv::gapi::imgproc::fluid::kernels());
        }
        else {
            throw std::logic_error("Unsupported kernel package type: " + type);
        }
        GAPI_Assert(false && "Unreachable code!");
    }

} // namespace util

struct smart_framing_filter {
	
	float threshold = 0.5f;

    std::shared_ptr<cv::GComputation> compute;
    cv::GCompileArgs compute_compile_args;

    // Use the media-io converter to both scale and convert the colorspace
    video_scaler_t* scalerToBGR = nullptr;
    video_scaler_t* scalerFromBGR = nullptr;

    std::string Device;
};

static const char* filter_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "OpenCV Smart Framing";
}


/**                   PROPERTIES                     */

static obs_properties_t* filter_properties(void* data)
{
	obs_properties_t* props = obs_properties_create();

	obs_property_t* p_inf_device = obs_properties_add_list(
		props,
		"Device",
		obs_module_text("Inference device"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_inf_device, obs_module_text("CPU"), DEVICE_CPU);
	obs_property_list_add_string(p_inf_device, obs_module_text("GPU"), DEVICE_GPU);
	obs_property_list_add_string(p_inf_device, obs_module_text("VPU"), DEVICE_VPU);

	UNUSED_PARAMETER(data);
	return props;
}

static void filter_defaults(obs_data_t* settings) {
	obs_data_set_default_string(settings, "Device", DEVICE_CPU);
}


/**                   FILTER CORE                     */
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(YOLOv4TinyNet, <GMat2(cv::GMat)>, "yolov4tiny_detector");

static std::shared_ptr<cv::GComputation> generate_smart_framing_graph()
{
    // Now build the graph
    cv::GMat in;
    cv::GMat out; //cropped resized output image
    cv::GMat out_sr; //cropped resized output image after super resolution
    cv::GMat out_sr_pp; //cropped resized output image after super resolution post processing
    cv::GMat blob26x26; //float32[1,255,26,26]
    cv::GMat blob13x13; //float32[1,255,13,13]
    cv::GArray<custom::DetectedObject> yolo_detections;
    cv::GArray<std::string> labels;
    std::tie(blob26x26, blob13x13) = cv::gapi::infer<YOLOv4TinyNet>(in);
    yolo_detections = custom::GYOLOv4TinyPostProcessingKernel::on(in, blob26x26, blob13x13, labels,
        0.5, //YOLO v4 Tiny confidence threshold.
        0.5, //YOLO v4 Tiny box IOU threshold.
        true); //use advanced post-processing for the YOLO v4 Tiny.

    if (false) 
    {
        out = custom::GSmartFramingMakeBorderKernel::on(in, yolo_detections);
    }
    else
    {
        out = custom::GSmartFramingKernel::on(in, yolo_detections);
    }

    return std::make_shared<cv::GComputation>(cv::GIn(in, labels), cv::GOut(cv::gapi::copy(in), yolo_detections, out, out));
}

static void filter_update(void* data, obs_data_t* settings)
{
    struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    const std::string current_device = obs_data_get_string(settings, "Device");

    if (!tf->compute || (tf->Device != current_device))
    {
        blog(LOG_INFO, "filter update: Creating new G-API Compute. Device = %s", tf->Device.c_str());
        tf->Device = current_device;
        tf->compute = generate_smart_framing_graph();
    }

}

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    smart_framing_filter* tf = new smart_framing_filter;

    char* model_env_val = std::getenv("OBS_SMART_FRAMING_MODEL_XML");
    if (model_env_val)
    {
        modelFilepathS = model_env_val;
    }

    blog(LOG_INFO, "IE model used = %s", modelFilepathS.c_str());

    /** Configure networks **/
	filter_update(tf, settings);

    blog(LOG_INFO, "<-filter_create");
	return tf;
}

static void destroyScalers(struct smart_framing_filter* tf) {
    blog(LOG_INFO, "Destroy scalers.");
    if (tf->scalerToBGR != nullptr) {
        video_scaler_destroy(tf->scalerToBGR);
        tf->scalerToBGR = nullptr;
    }
    if (tf->scalerFromBGR != nullptr) {
        video_scaler_destroy(tf->scalerFromBGR);
        tf->scalerFromBGR = nullptr;
    }
}

static void initializeScalers(
    cv::Size frameSize,
    enum video_format frameFormat,
    struct smart_framing_filter* tf
) {

    struct video_scale_info dst {
        VIDEO_FORMAT_BGR3,
            (uint32_t)frameSize.width,
            (uint32_t)frameSize.height,
            VIDEO_RANGE_DEFAULT,
            VIDEO_CS_DEFAULT
    };
    struct video_scale_info src {
        frameFormat,
            (uint32_t)frameSize.width,
            (uint32_t)frameSize.height,
            VIDEO_RANGE_DEFAULT,
            VIDEO_CS_DEFAULT
    };

    // Check if scalers already defined and release them
    destroyScalers(tf);

    blog(LOG_INFO, "Initialize scalers. Size %d x %d",
        frameSize.width, frameSize.height);

    // Create new scalers
    video_scaler_create(&tf->scalerToBGR, &dst, &src, VIDEO_SCALE_DEFAULT);
    video_scaler_create(&tf->scalerFromBGR, &src, &dst, VIDEO_SCALE_DEFAULT);
}


static cv::Mat convertFrameToBGR(
    struct obs_source_frame* frame,
    struct smart_framing_filter* tf
) {
    const cv::Size frameSize(frame->width, frame->height);

    if (tf->scalerToBGR == nullptr) {
        // Lazy initialize the frame scale & color converter
        initializeScalers(frameSize, frame->format, tf);
    }

    cv::Mat imageBGR(frameSize, CV_8UC3);
    const uint32_t bgrLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
    video_scaler_scale(tf->scalerToBGR,
        &(imageBGR.data), &(bgrLinesize),
        frame->data, frame->linesize);

    return imageBGR;
}

static void convertBGRToFrame(
    const cv::Mat& imageBGR,
    struct obs_source_frame* frame,
    struct smart_framing_filter* tf
) {
    if (tf->scalerFromBGR == nullptr) {
        // Lazy initialize the frame scale & color converter
        initializeScalers(cv::Size(frame->width, frame->height), frame->format, tf);
    }

    const uint32_t rgbLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
    video_scaler_scale(tf->scalerFromBGR,
        frame->data, frame->linesize,
        &(imageBGR.data), &(rgbLinesize));
}

static struct obs_source_frame* filter_render(void* data, struct obs_source_frame* frame)
{
	struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    // Convert to BGR
    cv::Mat imageBGR = convertFrameToBGR(frame, tf);

    cv::Mat image;
    cv::Mat out_image;
    cv::Mat out_image_sr;
    std::vector<custom::DetectedObject> objects;

    auto compute = tf->compute;

    //TODO: Move this stuff outside of filter_render()
    cv::gapi::GNetPackage networks;
    const auto net = cv::gapi::ie::Params<YOLOv4TinyNet>{
            modelFilepathS,                         // path to topology IR
            fileNameNoExt(modelFilepathS) + ".bin", // path to weights
            tf->Device }.cfgOutputLayers({ "conv2d_20/BiasAdd/Add", "conv2d_17/BiasAdd/Add" }).cfgInputLayers({ "image_input" });
    networks = cv::gapi::networks(net);

    /** Custom kernels plus CPU or Fluid **/
    auto kernels = cv::gapi::combine(custom::kernels(),
        util::getKernelPackage("opencv"));

    std::vector<std::string> coco_labels;

    try {
        compute->apply(cv::gin(imageBGR, coco_labels),
            cv::gout(image, objects, out_image, out_image_sr),
            cv::compile_args(kernels, networks));

        convertBGRToFrame(out_image, frame, tf);
    }
    catch (const std::exception& error) {
        blog(LOG_INFO, "in G-API apply method, exception: %s", error.what());
    }
    catch (...) {
        blog(LOG_INFO, " in G-API apply method Unknown/internal exception happened");
    }

    return frame;
}


static void filter_destroy(void* data)
{
	struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

	if (tf) {
        destroyScalers(tf);

        delete tf;
	}
}

struct obs_source_info smart_framing_filter_info_ocv = {
	.id = "smart_framing_ocv",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
	.get_name = filter_getname,
	.create = filter_create,
	.destroy = filter_destroy,
	.get_defaults = filter_defaults,
	.get_properties = filter_properties,
	.update = filter_update,
	.filter_video = filter_render,
};
