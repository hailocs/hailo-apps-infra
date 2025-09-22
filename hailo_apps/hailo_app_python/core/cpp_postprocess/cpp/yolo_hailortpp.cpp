#include <regex>
#include <fstream>
#include <sstream>
#include <map>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"
#include "json_config.hpp"
#include "common/labels/coco_eighty.hpp"
#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"

static const std::string DEFAULT_YOLOV5S_OUTPUT_LAYER = "yolov5s_nv12/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5m_wo_spp_60p/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER = "yolov5m_vehicles/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV8S_OUTPUT_LAYER = "yolov8s/yolov8_nms_postprocess";
static const std::string DEFAULT_YOLOV8M_OUTPUT_LAYER = "yolov8m/yolov8_nms_postprocess";
static const std::string DEFAULT_YOLOV8N_RELU6_LICENSE_PLATE_OUTPUT_LAYER = "yolov8n_relu6_license_plate/yolov8_nms_postprocess";

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

YoloParamsNMS *init(const std::string config_path, const std::string function_name)
{
    YoloParamsNMS *params;
    if (!fs::exists(config_path))
    {
        params = new YoloParamsNMS(common::coco_eighty);
        return params;
    }
    else
    {
        params = new YoloParamsNMS();
        char config_buffer[4096];
        const char *json_schema = R""""({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "detection_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "max_boxes": {
            "type": "integer"
            },
            "labels": {
            "type": "array",
            "items": {
                "type": "string"
                }
            }
        },
        "required": [
            "labels"
        ]
        })"""";

        std::FILE *fp = fopen(config_path.c_str(), "r");
        if (fp == nullptr)
        {
            throw std::runtime_error("JSON config file is not valid");
        }
        rapidjson::FileReadStream stream(fp, config_buffer, sizeof(config_buffer));
        bool valid = common::validate_json_with_schema(stream, json_schema);
        if (valid)
        {
            rapidjson::Document doc_config_json;
            doc_config_json.ParseStream(stream);

            // parse labels
            auto labels = doc_config_json["labels"].GetArray();
            uint i = 0;
            for (auto &v : labels)
            {
                params->labels.insert(std::pair<std::uint8_t, std::string>(i, v.GetString()));
                i++;
            }

            // set the params
            if (doc_config_json.HasMember("detection_threshold")) {
                params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            }
            if (doc_config_json.HasMember("max_boxes")) {
                params->max_boxes = doc_config_json["max_boxes"].GetInt();
                params->filter_by_score = true;
            }
        }
        fclose(fp);
    }
    return params;
}
void free_resources(void *params_void_ptr)
{
    YoloParamsNMS *params = reinterpret_cast<YoloParamsNMS *>(params_void_ptr);
    delete params;
}

static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};
static std::map<uint8_t, std::string> yolo_personface = {
        {0, "unlabeled"},
        {1, "person"},
        {2, "face"}};
static std::map<uint8_t, std::string> yolo_license_plate_labels = {
    {0, "license_plate"}};
void yolov5(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5s_nv12(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}
void yolov8n_relu6_license_plate(HailoROIPtr roi)
{
    std::cout << "\n=== YOLOv8n License Plate Debug ===" << std::endl;

    if (!roi->has_tensors()) {
        std::cout << "DEBUG: ROI has no tensors." << std::endl;
        return;
    }

    // List tensors
    std::cout << "DEBUG: ROI tensors available:" << std::endl;
    for (auto &tensor : roi->get_tensors()) {
        std::cout << "   -> " << tensor->name()
                  << " (size=" << tensor->size() << " bytes)" << std::endl;
    }

    // Get our output tensor
    auto tensor = roi->get_tensor(DEFAULT_YOLOV8N_RELU6_LICENSE_PLATE_OUTPUT_LAYER);
    if (!tensor) {
        std::cerr << "ERROR: Could not find output tensor: "
                  << DEFAULT_YOLOV8N_RELU6_LICENSE_PLATE_OUTPUT_LAYER << std::endl;
        return;
    }

    auto fmt = tensor->format();
    std::cout << "DEBUG: Tensor name: " << tensor->name() << std::endl;
    std::cout << "DEBUG: Tensor size (bytes): " << tensor->size()
              << " is_nms=" << fmt.is_nms << std::endl;

    // Dump first few raw values
    uint8_t *buffer = tensor->data();
    size_t buf_size = tensor->size();
    std::cout << "DEBUG: Dumping first 20 float values from tensor buffer:" << std::endl;
    for (int i = 0; i < 20 && (i * sizeof(float)) < buf_size; i++) {
        float val;
        memcpy(&val, buffer + i * sizeof(float), sizeof(val));
        std::cout << "   [" << i << "] " << val << std::endl;
    }

    // Decode
    try {
        auto post = HailoNMSDecode(tensor, yolo_license_plate_labels, 0.0f, 200, false);
        auto detections = post.decode_debug<float32_t, common::hailo_bbox_float32_t>();

        std::cout << "DEBUG: Total decoded detections = " << detections.size() << std::endl;
        for (size_t i = 0; i < detections.size(); i++) {
            auto &det = detections[i];
            auto bbox = det.get_bbox();
            std::cout << "   Detection[" << i << "] "
                      << "label=" << det.get_label()
                      << " conf=" << det.get_confidence()
                      << " bbox=(" << bbox.xmin() << ","
                                   << bbox.ymin() << ","
                                   << bbox.width() << ","
                                   << bbox.height() << ")"
                      << std::endl;
        }

        hailo_common::add_detections(roi, detections);
        std::cout << "DEBUG: Added detections into ROI." << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "ERROR: Exception during NMS decode: " << e.what() << std::endl;
    }

    std::cout << "=== End of YOLOv8n License Plate Debug ===\n" << std::endl;
}



void yolov8s(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov8m(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolox(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor("yolox_nms_postprocess"), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m_vehicles(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER), yolo_vehicles_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m_vehicles_nv12(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor("yolov5m_vehicles_nv12/yolov5_nms_postprocess"), yolo_vehicles_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5s_personface(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor("yolov5s_personface_nv12/yolov5_nms_postprocess"), yolo_personface);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5s_personface_rgb(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor("yolov5s_personface/yolov5_nms_postprocess"), yolo_personface);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    for (auto it = detections.begin(); it != detections.end();)
    {
        if (it->get_label() == "face")
        {
            it = detections.erase(it);
        }
        else
        {
            ++it;
        }
    }
    hailo_common::add_detections(roi, detections);
}

void yolov5_no_persons(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    for (auto it = detections.begin(); it != detections.end();)
    {
        // TODO
        if (it->get_label() == "person")
        {
            it = detections.erase(it);
        }
        else
        {
            ++it;
        }
    }
    hailo_common::add_detections(roi, detections);
}
void filter(HailoROIPtr roi, void *params_void_ptr)
{
    if (!roi->has_tensors())
    {
        return;
    }
    YoloParamsNMS *params = reinterpret_cast<YoloParamsNMS *>(params_void_ptr);
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    // find the nms tensor
    for (auto tensor : tensors)
    {
        if (std::regex_search(tensor->name(), std::regex("nms_postprocess"))) 
        {
            auto post = HailoNMSDecode(tensor, params->labels, params->detection_threshold, params->max_boxes, params->filter_by_score);
            auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
            hailo_common::add_detections(roi, detections);
        }
    }
}
void filter_letterbox(HailoROIPtr roi, void *params_void_ptr)
{
    filter(roi, params_void_ptr);
    // Resize Letterbox
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detections = hailo_common::get_hailo_detections(roi);
    for (auto &detection : detections)
    {
        auto detection_bbox = detection->get_bbox();
        auto xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        auto xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        HailoBBox new_bbox(xmin, ymin, xmax - xmin, ymax - ymin);
        detection->set_bbox(new_bbox);
    }

    // Clear the scaling bbox of main roi because all detections are fixed.
    roi->clear_scaling_bbox();

}