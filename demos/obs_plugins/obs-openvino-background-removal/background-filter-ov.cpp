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
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <models/segmentation_model.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <models/results.h>

//const char* MODEL_SINET = "SINet_Softmax_simple.onnx";
//const char* MODEL_MODNET = "modnet_simple.onnx";
//const char* MODEL_MEDIAPIPE = "mediapipe.onnx";
//const char* MODEL_SELFIE = "selfie_segmentation.onnx";
//const char* MODEL_RVM = "rvm_mobilenetv3_fp32.onnx";
const char* MODEL_DEEPLABV3 = "deeplabv3.xml";

const char* DEVICE_CPU = "CPU";
const char* DEVICE_VPU = "VPUX";
const char* DEVICE_GPU = "GPU";

InferenceEngine::Core core;
//const std::string modelFilepath = "D:\\open_model_zoo\\models\\public\\deeplabv3\\FP16\\deeplabv3.xml";


struct background_removal_filter {
	
    std::string modelFilepath;
	cv::Scalar backgroundColor{ 0, 0, 0 };
    int blur_value;
    bool blur;
	float contourFilter = 0.05f;
	float smoothContour = 0.5f;
	float feather = 0.0f;
    bool background_image;
    std::string background_image_path;
	std::string Device;
	std::string modelSelection;

	// Use the media-io converter to both scale and convert the colorspace
	video_scaler_t* scalerToBGR = nullptr;
	video_scaler_t* scalerFromBGR = nullptr;
	std::uint32_t nireq = 0;
	std::uint32_t nthreads = 0;
	std::string nstreams = "";
	std::string layout = "";
	bool auto_resize = false;

	cv::Mat backgroundMask;
	int maskEveryXFrames = 1;
	int maskEveryXFramesCount = 0;

    std::shared_ptr<AsyncPipeline> pipeline;

};


static const char* filter_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "OpenVINO Background Removal";
}


/**                   PROPERTIES                     */

static obs_properties_t* filter_properties(void* data)
{
	obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_path = obs_properties_add_path(
        props,
        "modelFilepath",
        obs_module_text("Inference Model Path"),
        OBS_PATH_FILE,
        ("*.xml"),
        "");


	obs_property_t* p_contour_filter = obs_properties_add_float_slider(
		props,
		"contour_filter",
		obs_module_text("Contour Filter (% of image)"),
		0.0,
		1.0,
		0.025);

	obs_property_t* p_smooth_contour = obs_properties_add_float_slider(
		props,
		"smooth_contour",
		obs_module_text("Smooth silhouette"),
		0.0,
		1.0,
		0.05);

	obs_property_t* p_feather = obs_properties_add_float_slider(
		props,
		"feather",
		obs_module_text("Feather blend silhouette"),
		0.0,
		1.0,
		0.05);

	obs_property_t* p_color = obs_properties_add_color(
		props,
		"replaceColor",
		obs_module_text("Background Color"));

    obs_property_t* p_background_image = obs_properties_add_bool(
        props,
        "AddCustomBackground",
        obs_module_text("Replace Background with custom image"));

    obs_property_t* p_background_image_path = obs_properties_add_path(
        props,
        "CustomImagePath",
        obs_module_text("Custom Background Image Path"),
        OBS_PATH_FILE,
        ("*.jpeg" " * .jpg"),
        "");

    obs_property_t* p_blur = obs_properties_add_bool(
        props,
        "blur_background",
        obs_module_text("Background Blur"));

    obs_property_t* p_blur_value = obs_properties_add_int(
        props,
        "blur_background_value",
        obs_module_text("Background Blur Intensity"),
        5,
        41,
        2);



	obs_property_t* p_inf_device = obs_properties_add_list(
		props,
		"Device",
		obs_module_text("Inference device"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_inf_device, obs_module_text("CPU"), DEVICE_CPU);
	obs_property_list_add_string(p_inf_device, obs_module_text("GPU"), DEVICE_GPU);
	obs_property_list_add_string(p_inf_device, obs_module_text("VPU"), DEVICE_VPU);


	obs_property_t* p_model_select = obs_properties_add_list(
		props,
		"model_select",
		obs_module_text("Segmentation model"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);


	obs_property_list_add_string(p_model_select, obs_module_text("Deeplabv3"), MODEL_DEEPLABV3);

	obs_property_t* p_mask_every_x_frames = obs_properties_add_int(
		props,
		"mask_every_x_frames",
		obs_module_text("Calculate mask every X frame"),
		1,
		300,
		1);

	UNUSED_PARAMETER(data);
	return props;
}

static void filter_defaults(obs_data_t* settings) {
    obs_data_set_default_string(settings, "modelFilepath", "C:\\Users\\arishaku\\deeplabv3\\FP16\\deeplabv3.xml"); //"C:\\Users\\arishaku\\deeplabv3\\FP16\\deeplabv3.xml" //"D:\\open_model_zoo\\models\\public\\deeplabv3\\FP16\\deeplabv3.xml"
	obs_data_set_default_double(settings, "contour_filter", 0.05);
	obs_data_set_default_double(settings, "smooth_contour", 0.5);
	obs_data_set_default_double(settings, "feather", 0.0);
	obs_data_set_default_int(settings, "replaceColor", 0x000000);
    obs_data_set_default_bool(settings, "AddCustomBackground", false);
    obs_data_set_default_string(settings, "CustomImagePath", "");
    obs_data_set_default_bool(settings, "blur_background", false);
    obs_data_set_default_int(settings, "blur_background_value", 21);
	obs_data_set_default_string(settings, "Device", DEVICE_CPU);
	obs_data_set_default_string(settings, "model_select", MODEL_DEEPLABV3);
	obs_data_set_default_int(settings, "mask_every_x_frames", 1);

}

static void destroyScalers(struct background_removal_filter* tf) {
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
    
	struct background_removal_filter* tf = reinterpret_cast<background_removal_filter*>(data);

    tf->modelFilepath = obs_data_get_string(settings, "modelFilepath");


	uint64_t color = obs_data_get_int(settings, "replaceColor");
	tf->backgroundColor.val[0] = (double)((color >> 16) & 0x0000ff);
	tf->backgroundColor.val[1] = (double)((color >> 8) & 0x0000ff);
	tf->backgroundColor.val[2] = (double)(color & 0x0000ff);

    tf->background_image = obs_data_get_bool(settings, "AddCustomBackground");
    tf->background_image_path = obs_data_get_string(settings, "CustomImagePath");

    tf->blur = obs_data_get_bool(settings, "blur_background");
    tf->blur_value = (int)obs_data_get_int(settings, "blur_background_value");

	tf->contourFilter = (float)obs_data_get_double(settings, "contour_filter");
	tf->smoothContour = (float)obs_data_get_double(settings, "smooth_contour");
	tf->feather = (float)obs_data_get_double(settings, "feather");
	tf->maskEveryXFrames = (int)obs_data_get_int(settings, "mask_every_x_frames");
	tf->maskEveryXFramesCount = (int)(0);


	const std::string current_device = obs_data_get_string(settings, "Device");
	const std::string newModel = obs_data_get_string(settings, "model_select");

	if (tf->modelSelection.empty() ||
		tf->modelSelection != newModel)
	{
		// Re-initialize model if it's not already the selected one or switching inference device
		tf->modelSelection = newModel;
		//tf->Device = newUseGpu;
		destroyScalers(tf);


	}

    try {
        if (!tf->pipeline || (tf->Device != current_device))
        {
            tf->Device = current_device;
            blog(LOG_INFO, "updating pipeline to use %s", tf->Device.c_str());
            tf->pipeline = std::make_shared<AsyncPipeline>(std::unique_ptr<SegmentationModel>(new SegmentationModel(tf->modelFilepath, false)),
                ConfigFactory::getUserConfig(tf->Device, "", "", false, 1, "", 0),
                core);
        }
    }

    catch
        (const std::exception& e) {
        blog(LOG_ERROR, "%s", e.what());
        tf->pipeline.reset();
        return;
    }
}


/**                   FILTER CORE                     */

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    background_removal_filter* tf = new background_removal_filter;


	std::string instanceName{ "background-removal-inference-ov" };
	

	tf->modelSelection = MODEL_DEEPLABV3;

   
    filter_update(tf, settings);
   
   

	return tf;
}


static void initializeScalers(
	cv::Size frameSize,
	enum video_format frameFormat,
	struct background_removal_filter* tf
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
	struct background_removal_filter* tf
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
	struct background_removal_filter* tf
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



static void processImageForBackground(
	struct background_removal_filter* tf,
	const cv::Mat& imageBGR,
	cv::Mat& backgroundMask)
{   
	//blog(LOG_INFO,"OpenVino version", ov::get_openvino_version());

	std::unique_ptr<ResultBase> result;

    auto pipeline = tf->pipeline;

    if (!pipeline)
    {
        blog(LOG_INFO, "Pipeline not valid");
        backgroundMask = cv::Mat::zeros(imageBGR.size(), CV_8UC1);
        return;
    }

	try {
		// To RGB
		cv::Mat imageRGB;
		cv::cvtColor(imageBGR, imageRGB, cv::COLOR_BGR2RGB);
		if (pipeline->isReadyToProcess()) {
			auto startTime = std::chrono::steady_clock::now();

            pipeline->submitData(ImageInputData(imageRGB),
				std::make_shared<ImageMetaData>(imageRGB, startTime));

	}
		else {
			blog(LOG_INFO, "Pipeline not ready");
		}

		//pipeline.waitForData();

        pipeline->waitForTotalCompletion();
		result = pipeline->getResult();
		if (!result) {
			blog(LOG_INFO, "is it valid result?");
			backgroundMask = cv::Mat::zeros(imageBGR.size(), CV_8UC1);
			return;

		}

		cv::Mat outputImage = result->asRef<ImageResult>().resultImage;


        backgroundMask = outputImage != 15 ; 


	
		// Contour processing
		if (tf->contourFilter > 0.0 && tf->contourFilter < 1.0) {
			std::vector<std::vector<cv::Point> > contours;
			findContours(backgroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			std::vector<std::vector<cv::Point> > filteredContours;
			const int64_t contourSizeThreshold = (int64_t)(backgroundMask.total() * tf->contourFilter);
			for (auto& contour : contours) {
				if (cv::contourArea(contour) > contourSizeThreshold) {
					filteredContours.push_back(contour);
				}
			}
			backgroundMask.setTo(0);
			drawContours(backgroundMask, filteredContours, -1, cv::Scalar(255), -1);
		}

		// Resize the size of the mask back to the size of the original input.
		cv::resize(backgroundMask, backgroundMask, imageBGR.size());

		// Smooth mask with a fast filter (box).
		if (tf->smoothContour > 0.0) {
			int k_size = (int)(100 * tf->smoothContour);
			cv::boxFilter(backgroundMask, backgroundMask, backgroundMask.depth(), cv::Size(k_size, k_size));
			backgroundMask = backgroundMask > 128;
		}
	}

	
	catch (const std::exception& e) {
		blog(LOG_ERROR, "%s", e.what());
	}
}



static struct obs_source_frame* filter_render(void* data, struct obs_source_frame* frame)
{
	struct background_removal_filter* tf = reinterpret_cast<background_removal_filter*>(data);

	// Convert to BGR
	cv::Mat imageBGR = convertFrameToBGR(frame, tf);


	cv::Mat backgroundMask(imageBGR.size(), CV_8UC1, cv::Scalar(255));

	tf->maskEveryXFramesCount = ++(tf->maskEveryXFramesCount) % tf->maskEveryXFrames;
	if (tf->maskEveryXFramesCount != 0 && !tf->backgroundMask.empty()) {
		// We are skipping processing of the mask for this frame.
		// Get the background mask previously generated.
		tf->backgroundMask.copyTo(backgroundMask);
	}
	else {
		// Process the image to find the mask.
		processImageForBackground(tf, imageBGR, backgroundMask);
  

		// Now that the mask is completed, save it off so it can be used on a later frame
		// if we've chosen to only process the mask every X frames.
		backgroundMask.copyTo(tf->backgroundMask);
        

	}
    cv::Mat alpha_im;
	// Apply the mask back to the main image.
	try {
		if ((tf->feather > 0.0) && (!tf->blur) && (!tf->background_image)) {
			// If we're going to feather/alpha blend, we need to do some processing that
			// will combine the blended "foreground" and "masked background" images onto the main image.
			cv::Mat maskFloat;
			int k_size = (int)(40 * tf->feather);

			// Convert Mat to float and Normalize the alpha mask to keep intensity between 0 and 1.
			backgroundMask.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);
			//Feather the normalized mask.
			cv::boxFilter(maskFloat, maskFloat, maskFloat.depth(), cv::Size(k_size, k_size));

			// Alpha blend
			cv::Mat maskFloat3c;
			cv::cvtColor(maskFloat, maskFloat3c, cv::COLOR_GRAY2BGR);
			cv::Mat tmpImage, tmpBackground;
			// Mutiply the unmasked foreground area of the image with ( 1 - alpha matte).
			cv::multiply(imageBGR, cv::Scalar(1, 1, 1) - maskFloat3c, tmpImage, 1.0, CV_32FC3);
			// Multiply the masked background area (with the background color applied) with the alpha matte.


			cv::multiply(cv::Mat(imageBGR.size(), CV_32FC3, tf->backgroundColor), maskFloat3c, tmpBackground);
			// Add the foreground and background images together, rescale back to an 8bit integer image
			// and apply onto the main image.
			cv::Mat(tmpImage + tmpBackground).convertTo(imageBGR, CV_8UC3);
		}
		else {
		
            if (tf->blur) {
                cv::Mat bg;
                blur(imageBGR, bg, cv::Size(tf->blur_value, tf->blur_value));
                for (int row = 0; row < backgroundMask.rows; ++row)
                {
                    for (int col = 0; col < backgroundMask.cols; ++col) {

                           if (backgroundMask.at<uchar>(row, col)){
                           
                              imageBGR.at<cv::Vec3b>(row, col) = bg.at<cv::Vec3b>(row, col);
                             
                           }
                    }
                }
            }
            else if ((tf->background_image) && (!tf->blur))
            {
                cv::Mat customImage = cv::imread(tf->background_image_path);
                cv::resize(customImage, customImage, imageBGR.size());
                for (int row = 0; row < backgroundMask.rows; ++row)
                {
                    for (int col = 0; col < backgroundMask.cols; ++col) {

                        if (backgroundMask.at<uchar>(row, col)) {

                            imageBGR.at<cv::Vec3b>(row, col) = customImage.at<cv::Vec3b>(row, col);
                        }
                    }
                }


            }
            else
			imageBGR.setTo(tf->backgroundColor, backgroundMask);
		}
	}
	catch (const std::exception& e) {
		blog(LOG_ERROR, "%s", e.what());
	}

	// Put masked image back on frame,

 

	convertBGRToFrame(imageBGR, frame, tf);
	return frame;
}


static void filter_destroy(void* data)
{
	struct background_removal_filter* tf = reinterpret_cast<background_removal_filter*>(data);

	if (tf) {
		destroyScalers(tf);
		//bfree(tf);
        delete tf;
	}
}



struct obs_source_info background_removal_filter_info_ov = {
	.id = "background_removal_ov",
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
