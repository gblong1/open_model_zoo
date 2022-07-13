#include <obs-module.h>
#include <media-io/video-scaler.h>

#ifdef _WIN32
#include <wchar.h>
#endif

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>

#include "plugin-macros.generated.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

//#include <gflags/gflags.h>


#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include "face_inference_results.hpp"

#include "face_detector.hpp"

#include "base_estimator.hpp"
#include "head_pose_estimator.hpp"
#include "landmarks_estimator.hpp"
#include "eye_state_estimator.hpp"
#include "gaze_estimator.hpp"

#include "results_marker.hpp"
#include "exponential_averager.hpp"

#include "utils.hpp"

const char* MODEL_FACEDETECT = "face-detection-retail-0004.xml";
const char* MODEL_FACIALLM = "facial-landmarks-35-adas-0002.xml";
const char* MODEL_HEADPOSE = "head-pose-estimation-adas-0001.xml";
const char* MODEL_EYESTATE = "open-closed-eye-0001.xml";
const char* MODEL_GAZESTIMATION = "gaze-estimation-adas-0002.xml";

using namespace gaze_estimation;

InferenceEngine::Core core;

struct gaze_correction_filter {
	
    //std::string modelFilepath;
	
	float threshold = 0.5f;
	
	std::string Device;

    std::shared_ptr<FaceDetector> faceDetector;
    std::shared_ptr<ResultsMarker> resultsMarker;
    std::shared_ptr<HeadPoseEstimator> headPoseEstimator;
    std::shared_ptr<LandmarksEstimator> landmarksEstimator;
    std::shared_ptr<EyeStateEstimator> eyeStateEstimator;
    std::shared_ptr<GazeEstimator> gazeEstimator;
    bool fd;
    bool lm;
    bool hp;
    bool es;
    bool gv;



    //std::shared_ptr<BaseEstimator> estimators = {};
   
   

	// Use the media-io converter to both scale and convert the colorspace
	video_scaler_t* scalerToBGR = nullptr;
	video_scaler_t* scalerFromBGR = nullptr;
	std::uint32_t nireq = 0;
	
    std::vector<std::string> ov_available_devices;
    //std::vector<FaceInferenceResults> inferenceResults;
   

};


static const char* filter_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "OpenVINO Gaze Correction";
}


/**                   PROPERTIES                     */

static obs_properties_t* filter_properties(void* data)
{
    struct gaze_correction_filter* tf = reinterpret_cast<gaze_correction_filter*>(data);
	obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_path = obs_properties_add_path(
        props,
        "modelFilepath",
        obs_module_text("Inference Model Path"),
        OBS_PATH_FILE,
        ("*.xml"),
        "");

    obs_property_t* p_FaceDetectorBB = obs_properties_add_bool(
        props,
        "FaceDetectorBB",
        obs_module_text("Face Detector BB"));

    obs_property_t* p_HeadPose = obs_properties_add_bool(
        props,
        "HeadPose",
        obs_module_text("Head Pose Vector"));

    obs_property_t* p_FacialLandmarks = obs_properties_add_bool(
        props,
        "FacialLandmarks",
        obs_module_text("Facial Landmarks Marker"));

    obs_property_t* p_EyeState = obs_properties_add_bool(
        props,
        "EyeState",
        obs_module_text("Eye State"));
    obs_property_t* p_GazeVector = obs_properties_add_bool(
        props,
        "GazeVector",
        obs_module_text("Gaze Vector"));


	obs_property_t* p_inf_device = obs_properties_add_list(
		props,
		"Device",
		obs_module_text("Inference device"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);

    for (auto device : tf->ov_available_devices)
    {
        obs_property_list_add_string(p_inf_device, obs_module_text(device.c_str()), device.c_str());
    }

	UNUSED_PARAMETER(data);
	return props;
}

static void filter_defaults(obs_data_t* settings) {
    obs_data_set_default_string(settings, "modelFilepath", "E:\\open_model_zoo\\models\\intel\\gaze-estimation-adas-0002\\FP16-INT8\\gaze-estimation-adas-0002.xml"); //"C:\\Users\\arishaku\\deeplabv3\\FP16\\deeplabv3.xml" //"D:\\open_model_zoo\\models\\public\\deeplabv3\\FP16\\deeplabv3.xml"
    obs_data_set_default_bool(settings, "FaceDetectorBB", false);
    obs_data_set_default_bool(settings, "HeadPose", false);
    obs_data_set_default_bool(settings, "FacialLandmarks", false);
    obs_data_set_default_bool(settings, "EyeState", true);
    obs_data_set_default_bool(settings, "GazeVector", true);
	obs_data_set_default_string(settings, "Device", "CPU");
}

static void destroyScalers(struct gaze_correction_filter* tf) {
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


static void filter_update(void* data, obs_data_t* settings)
{
    
	struct gaze_correction_filter* tf = reinterpret_cast<gaze_correction_filter*>(data);

    //tf->modelFilepath = obs_data_get_string(settings, "modelFilepath");


	const std::string current_device = obs_data_get_string(settings, "Device");
    tf->Device = current_device;

    std::string modelFilepath = "E:\\open_model_zoo\\models\\intel\\gaze-estimation-adas-0002\\FP16-INT8\\gaze-estimation-adas-0002.xml"; //TODO: figure out best way to call model file path 
    std::string m_fd = "E:\\open_model_zoo\\models\\intel\\face-detection-retail-0004\\FP16-INT8\\face-detection-retail-0004.xml";
    std::string m_hp = "E:\\open_model_zoo\\models\\intel\\head-pose-estimation-adas-0001\\FP16-INT8\\head-pose-estimation-adas-0001.xml";
    std::string m_lm = "E:\\open_model_zoo\\models\\intel\\facial-landmarks-35-adas-0002\\FP16-INT8\\facial-landmarks-35-adas-0002.xml";
    std::string m_es = "E:\\open_model_zoo\\models\\public\\open-closed-eye-0001\\FP16\\open-closed-eye-0001.xml";

    char* ge_model_path = obs_module_file("gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml");
    if (ge_model_path)
    {
        modelFilepath = ge_model_path;
    }

    char* fd_model_path = obs_module_file("face-detection-retail-0004/FP16-INT8/face-detection-retail-0004.xml");
    if (fd_model_path)
    {
        m_fd = fd_model_path;
    }

    char* hp_model_path = obs_module_file("head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml");
    if (hp_model_path)
    {
        m_hp = hp_model_path;
    }

    char* lm_model_path = obs_module_file("facial-landmarks-35-adas-0002/FP16-INT8/facial-landmarks-35-adas-0002.xml");
    if (lm_model_path)
    {
        m_lm = lm_model_path;
    }

    char* es_model_path = obs_module_file("open-closed-eye-0001/FP16/open-closed-eye-0001.xml");
    if (es_model_path)
    {
        m_es = es_model_path;
    }

    try {
        // Set up face detector and estimators

        tf->fd = obs_data_get_bool(settings, "FaceDetectorBB");
        tf->hp = obs_data_get_bool(settings, "HeadPose");
        tf->lm = obs_data_get_bool(settings, "FacialLandmarks");
        tf->es = obs_data_get_bool(settings, "EyeState");
        tf->gv = obs_data_get_bool(settings, "GazeVector");

        InferenceEngine::Core ie;
        tf->faceDetector= std::make_shared<FaceDetector>(ie, m_fd, tf->Device, tf->threshold, false);
        
        tf->headPoseEstimator = std::make_shared<HeadPoseEstimator>(ie, m_hp, tf->Device);
        tf->landmarksEstimator = std::make_shared<LandmarksEstimator>(ie, m_lm, tf->Device);
        tf->eyeStateEstimator = std::make_shared<EyeStateEstimator>(ie, m_es, tf->Device);
        tf->gazeEstimator = std::make_shared<GazeEstimator>(ie, modelFilepath, tf->Device);

        tf->resultsMarker = std::make_shared<ResultsMarker>(tf->fd, tf->hp, tf->lm, tf->gv, tf->es);

    }

    catch
        (const std::exception& e) {
        blog(LOG_ERROR, "%s", e.what());
        
        return;
    }
}


/**                   FILTER CORE                     */

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    gaze_correction_filter* tf = new gaze_correction_filter;

    tf->ov_available_devices = core.GetAvailableDevices();
    if (tf->ov_available_devices.empty())
    {
        blog(LOG_INFO, "No available OpenVINO devices found.");
        delete tf;
        return NULL;
    }

    filter_update(tf, settings);

	return tf;
}


static void initializeScalers(
	cv::Size frameSize,
	enum video_format frameFormat,
	struct gaze_correction_filter* tf
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
	struct gaze_correction_filter* tf
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
	struct gaze_correction_filter* tf
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
	struct gaze_correction_filter* tf = reinterpret_cast<gaze_correction_filter*>(data);

	// Convert to BGR
	cv::Mat imageBGR = convertFrameToBGR(frame, tf);
  
    BaseEstimator* estimators[] = { tf->headPoseEstimator.get(),tf->landmarksEstimator.get(), tf->eyeStateEstimator.get(), tf->gazeEstimator.get() };

    try {
        
        auto inferenceResults = tf->faceDetector->detect(imageBGR);
        
        for (auto& inferenceResult : inferenceResults) {
            for (auto estimator : estimators) {
               
                estimator->estimate(imageBGR, inferenceResult);
            }
        }

        // Display the results
     
        for (auto const& inferenceResult : inferenceResults) {
            tf->resultsMarker->mark(imageBGR, inferenceResult);
        }
    }
   catch (const std::exception& e) {  
        blog(LOG_ERROR, "%s", e.what());

        return frame;
    }
    

	// Put overlay image back on frame
	convertBGRToFrame(imageBGR, frame, tf);
	return frame;
}


static void filter_destroy(void* data)
{
	struct gaze_correction_filter* tf = reinterpret_cast<gaze_correction_filter*>(data);

	if (tf) {
		destroyScalers(tf);
		//bfree(tf);
        delete tf;
	}
}



struct obs_source_info gaze_correction_filter_info_ov = {
	.id = "obs-gaze-correction-ov",
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
