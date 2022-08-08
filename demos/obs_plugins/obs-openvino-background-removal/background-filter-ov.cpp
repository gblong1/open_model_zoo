#include <obs-module.h>
#include <media-io/video-scaler.h>

#ifdef _WIN32
#include <wchar.h>
#endif

#include <opencv2/imgproc.hpp>

#include <ittutils.h>

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


struct ModelDetails
{
    std::string data_path; //path to xml
    InferenceEngine::Precision input_tensor_precision;
    bool bInputAsBGR; //true means model expects BGR input, false means model expects RGB
};

const std::map<const std::string, ModelDetails> modelname_to_datapath
{
    {
        "DeepLabv3",
        {
            "deeplabv3/FP16/deeplabv3.xml",
            InferenceEngine::Precision::U8,
            true
        }
    },
    {
        "SelfieSegmentation(MediaPipe)",
        {
            "mediapipe_self_seg/FP16/mediapipe_selfie_segmentation.xml",
            InferenceEngine::Precision::FP32,
            false
        }
    }
};

InferenceEngine::Core core;

struct background_removal_filter {
	
	cv::Scalar backgroundColor{ 0, 0, 0 };
    int blur_value;
    bool blur;
	float smoothContour = 0.5f;
	float feather = 0.0f;
    bool background_image;
    bool bConvertToRGB = false;
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

    std::vector<std::string> ov_available_devices;

    std::shared_ptr<AsyncPipeline> pipeline;

    std::shared_ptr<cv::Mat> customImage;

};


static const char* filter_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "OpenVINO Background Removal";
}


/**                   PROPERTIES                     */

static obs_properties_t* filter_properties(void* data)
{
    struct background_removal_filter* tf = reinterpret_cast<background_removal_filter*>(data);

	obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_select = obs_properties_add_list(
        props,
        "model_select",
        obs_module_text("Segmentation model"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto modentry : modelname_to_datapath)
    {
        obs_property_list_add_string(p_model_select, obs_module_text(modentry.first.c_str()), modentry.first.c_str());
    }

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

    for (auto device : tf->ov_available_devices)
    {
        obs_property_list_add_string(p_inf_device, obs_module_text(device.c_str()), device.c_str());
    }

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
	obs_data_set_default_double(settings, "smooth_contour", 0.5);
	obs_data_set_default_double(settings, "feather", 0.0);
	obs_data_set_default_int(settings, "replaceColor", 0x000000);
    obs_data_set_default_bool(settings, "AddCustomBackground", false);
    obs_data_set_default_string(settings, "CustomImagePath", "");
    obs_data_set_default_bool(settings, "blur_background", false);
    obs_data_set_default_int(settings, "blur_background_value", 21);
	obs_data_set_default_string(settings, "Device", "CPU");
	obs_data_set_default_string(settings, "model_select",
        modelname_to_datapath.begin()->first.c_str());
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

	uint64_t color = obs_data_get_int(settings, "replaceColor");
	tf->backgroundColor.val[0] = (double)((color >> 16) & 0x0000ff);
	tf->backgroundColor.val[1] = (double)((color >> 8) & 0x0000ff);
	tf->backgroundColor.val[2] = (double)(color & 0x0000ff);

    tf->background_image = obs_data_get_bool(settings, "AddCustomBackground");

    std::string bkg_img_path = obs_data_get_string(settings, "CustomImagePath");
    if (tf->background_image && (tf->background_image_path != bkg_img_path))
    {
        tf->background_image_path = bkg_img_path;
        tf->customImage = std::make_shared<cv::Mat>(cv::imread(tf->background_image_path));
    }

    tf->blur = obs_data_get_bool(settings, "blur_background");
    tf->blur_value = (int)obs_data_get_int(settings, "blur_background_value");

	tf->smoothContour = (float)obs_data_get_double(settings, "smooth_contour");
	tf->feather = (float)obs_data_get_double(settings, "feather");
	tf->maskEveryXFrames = (int)obs_data_get_int(settings, "mask_every_x_frames");
	tf->maskEveryXFramesCount = (int)(0);

	const std::string current_device = obs_data_get_string(settings, "Device");
	const std::string newModel = obs_data_get_string(settings, "model_select");

    if (!tf->pipeline || (tf->Device != current_device) || (tf->modelSelection != newModel))
    {
        destroyScalers(tf);
        tf->Device = current_device;
        tf->modelSelection = newModel;
        
        ModelDetails details = modelname_to_datapath.at(newModel);
        char* model_file_path = obs_module_file(details.data_path.c_str());

        tf->bConvertToRGB = !details.bInputAsBGR;

        if (model_file_path)
        {
            //create the pipeline
            try {
                ITT_SCOPED_TASK(bkg_rm_create_async_pipeline)
                    blog(LOG_INFO, "updating pipeline to use model=%s, running on device=%s", details.data_path.c_str(), tf->Device.c_str());
                tf->pipeline = std::make_shared<AsyncPipeline>(std::unique_ptr<SegmentationModel>(new SegmentationModel(model_file_path, false, details.input_tensor_precision)),
                    ConfigFactory::getUserConfig(tf->Device, "", "", false, 1, "", 0),
                    core);
            }
            catch (const std::exception& e) {
                blog(LOG_ERROR, "%s", e.what());
                tf->pipeline.reset();
                return;
            }
        }
        else
        {
            blog(LOG_ERROR, "Unable to find model file, %s, in obs-studio plugin module directory", details.data_path.c_str());
        }  
    }
}


/**                   FILTER CORE                     */

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    background_removal_filter* tf = new background_removal_filter;


	std::string instanceName{ "background-removal-inference-ov" };
	
    tf->ov_available_devices = core.GetAvailableDevices();
    if (tf->ov_available_devices.empty())
    {
        blog(LOG_ERROR, "No available OpenVINO devices found.");
        delete tf;
        return NULL;
    }

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
    ITT_SCOPED_TASK(bkg_rm_convertFrameToBGR)
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
    ITT_SCOPED_TASK(bkg_rm_convertBGRToFrame)
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
    ITT_SCOPED_TASK(processImageForBackground)
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
        {
            ITT_SCOPED_TASK(RunAsyncPipeline)
            if (pipeline->isReadyToProcess()) {
                auto startTime = std::chrono::steady_clock::now();

                if (tf->bConvertToRGB)
                {
                    cv::Mat imageRGB;
                    cv::cvtColor(imageBGR, imageRGB, cv::COLOR_BGR2RGB);
                    pipeline->submitData(ImageInputData(imageRGB),
                        std::make_shared<ImageMetaData>(imageRGB, startTime));
                }
                else
                {
                    pipeline->submitData(ImageInputData(imageBGR),
                        std::make_shared<ImageMetaData>(imageBGR, startTime));
                }
                

            }
            else {
                blog(LOG_INFO, "Pipeline not ready");
            }
            pipeline->waitForTotalCompletion();
        }
		result = pipeline->getResult();
		if (!result) {
			blog(LOG_INFO, "is it valid result?");
			backgroundMask = cv::Mat::zeros(imageBGR.size(), CV_8UC1);
			return;

		}

		cv::Mat outputImage = result->asRef<ImageResult>().resultImage;

        backgroundMask = outputImage != 15 ; 

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
    ITT_SCOPED_TASK(bkg_rm_filter_render);
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

            if(tf->background_image || tf->blur)
            {
                if (tf->background_image)
                {
                    auto bkg_image = tf->customImage;
                    if (bkg_image)
                    {
                        cv::Mat customImage;
                        cv::resize(*bkg_image, customImage, imageBGR.size());
                        for (int row = 0; row < backgroundMask.rows; ++row)
                        {
                            for (int col = 0; col < backgroundMask.cols; ++col) {

                                if (backgroundMask.at<uchar>(row, col)) {

                                    imageBGR.at<cv::Vec3b>(row, col) = customImage.at<cv::Vec3b>(row, col);
                                }
                            }
                        }
                    }
                }

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

            }
            else
            { 
			    imageBGR.setTo(tf->backgroundColor, backgroundMask);
            }
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
