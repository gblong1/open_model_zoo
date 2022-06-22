#include <openvino/openvino.hpp>
#include <obs-module.h>
#include <media-io/video-scaler.h>

#ifdef _WIN32
#include <wchar.h>
#endif

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>

#include "plugin-macros.generated.h"

struct ov_background_removal_filter {
	float threshold = 0.5f;
#if _WIN32
	const wchar_t* modelFilepath = nullptr;
#else
	const char* modelFilepath = nullptr;
#endif
};

static const char *filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "OpenVINO Background Removal";
}

/**                   PROPERTIES                     */
static obs_properties_t *filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *p_threshold = obs_properties_add_float_slider(
		props,
		"threshold",
		obs_module_text("Threshold"),
		0.0,
		1.0,
		0.025);

	UNUSED_PARAMETER(data);
	return props;
}

static void filter_defaults(obs_data_t *settings) {
	obs_data_set_default_double(settings, "threshold", 0.5);
}

static void filter_update(void *data, obs_data_t *settings)
{
	struct ov_background_removal_filter *tf = reinterpret_cast<ov_background_removal_filter*>(data);
	tf->threshold = (float)obs_data_get_double(settings, "threshold");
}

/**                   FILTER CORE                     */

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct ov_background_removal_filter*tf = reinterpret_cast<ov_background_removal_filter*>(bzalloc(sizeof(struct ov_background_removal_filter)));

	std::string instanceName{"openvino-background-removal-inference"};

	filter_update(tf, settings);

	return tf;
}

static struct obs_source_frame * filter_render(void *data, struct obs_source_frame *frame)
{
	struct ov_background_removal_filter *tf = reinterpret_cast<ov_background_removal_filter*>(data);

	return frame;
}

static void filter_destroy(void *data)
{
	struct ov_background_removal_filter *tf = reinterpret_cast<ov_background_removal_filter *>(data);

	if (tf) {
		bfree(tf);
	}
}

struct obs_source_info ov_background_removal_filter_info = {
	.id = "openvino_background_removal",
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
