#include "ocr_postprocess.hpp"  
  
#include <cstdio>  
#include <cstring>  
#include <cmath>  
#include <fstream>  
#include <algorithm>  
#include <numeric>  
#include <stdexcept>  
#include <iostream>  
  
#include <opencv2/imgproc.hpp>  
#include <opencv2/imgcodecs.hpp>  
  
// RapidJSON for optional config  
#include "rapidjson/document.h"  
#include "rapidjson/error/en.h"  
#include "rapidjson/filereadstream.h"  
#include "rapidjson/schema.h"  
  
#if __GNUC__ > 8  
  #include <filesystem>  
  namespace fs = std::filesystem;  
#else  
  #include <experimental/filesystem>  
  namespace fs = std::experimental::filesystem;  
#endif  

// ---------------------------  
// JSON helpers  
// ---------------------------  
static bool validate_json_with_schema(FILE *fp, const char *schema) {  
    char buffer[1 << 12];  
    rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));  
  
    rapidjson::Document sd; sd.Parse(schema);  
    if (sd.HasParseError()) return false;  
  
    rapidjson::SchemaDocument sdoc(sd);  
    rapidjson::SchemaValidator validator(sdoc);  
  
    fseek(fp, 0, SEEK_SET);  
    rapidjson::FileReadStream is2(fp, buffer, sizeof(buffer));  
    rapidjson::Document d; d.ParseStream(is2);  
    if (d.HasParseError()) return false;  
  
    return d.Accept(validator);  
}  
  
static void load_default_charset(OcrParams &p) {  
    // Based on your Python CHARACTERS array - add blank first, then 0-9, symbols, A-Z, a-z  
    p.charset.emplace_back("blank");  // blank token at index 0  
    for (char c='0'; c<='9'; ++c) p.charset.emplace_back(1, c);  
    // Add common symbols found in license plates  
    p.charset.emplace_back(":");  
    p.charset.emplace_back(";");  
    p.charset.emplace_back("<");  
    p.charset.emplace_back("=");  
    p.charset.emplace_back(">");  
    p.charset.emplace_back("?");  
    p.charset.emplace_back("@");  
    for (char c='A'; c<='Z'; ++c) p.charset.emplace_back(1, c);  
    p.charset.emplace_back("[");  
    p.charset.emplace_back("\\");  
    p.charset.emplace_back("]");  
    p.charset.emplace_back("^");  
    p.charset.emplace_back("_");  
    p.charset.emplace_back("`");  
    for (char c='a'; c<='z'; ++c) p.charset.emplace_back(1, c);  
    p.charset.emplace_back("{");  
    p.charset.emplace_back("|");  
    p.charset.emplace_back("}");  
    p.charset.emplace_back("~");  
    p.charset.emplace_back("!");  
    p.charset.emplace_back("\"");  
    p.charset.emplace_back("#");  
    p.charset.emplace_back("$");  
    p.charset.emplace_back("%");  
    p.charset.emplace_back("&");  
    p.charset.emplace_back("'");  
    p.charset.emplace_back("(");  
    p.charset.emplace_back(")");  
    p.charset.emplace_back("*");  
    p.charset.emplace_back("+");  
    p.charset.emplace_back(",");  
    p.charset.emplace_back("-");  
    p.charset.emplace_back(".");  
    p.charset.emplace_back("/");  
    p.charset.emplace_back(" ");  
}  
  
static void load_charset_from_file(OcrParams &p) {  
    if (p.charset_path.empty()) { load_default_charset(p); return; }  
    std::ifstream in(p.charset_path);  
    if (!in.is_open()) throw std::runtime_error("Failed to open charset file: " + p.charset_path);  
    std::string line;  
    while (std::getline(in, line)) p.charset.push_back(line);  
    if (p.charset.empty()) load_default_charset(p);  
}  
  
// ---------------------------  
// init / free_resources  
// ---------------------------  
OcrParams *init(const std::string config_path, const std::string /*function_name*/) {  
    auto *params = new OcrParams();  
  
    if (fs::exists(config_path)) {  
        const char *schema = R""""({  
          "$schema": "http://json-schema.org/draft-04/schema#",  
          "type": "object",  
          "properties": {  
            "det_bin_thresh":     { "type": "number" },  
            "det_box_thresh":     { "type": "number" },  
            "det_unclip_ratio":   { "type": "number" },  
            "det_max_candidates": { "type": "integer" },  
            "det_min_box_size":   { "type": "number" },  
            "det_output_name":    { "type": "string" },  
            "det_map_h":          { "type": "integer" },  
            "det_map_w":          { "type": "integer" },  
            "letterbox_fix":      { "type": "boolean" },  
  
            "rec_output_name":    { "type": "string" },  
            "charset_path":       { "type": "string" },  
            "blank_index":        { "type": "integer" },  
            "logits_are_softmax": { "type": "boolean" },  
            "time_major":         { "type": "boolean" },  
            "text_conf_smooth":   { "type": "number" },  
            "attach_caption_box": { "type": "boolean" }  
          }  
        })"""";  
  
        FILE *fp = fopen(config_path.c_str(), "r");  
        if (!fp) throw std::runtime_error("JSON config file cannot be opened");  
        bool ok = validate_json_with_schema(fp, schema);  
        if (!ok) { fclose(fp); throw std::runtime_error("JSON config doesn't match schema"); }  
        fseek(fp, 0, SEEK_SET);  
  
        char buffer[1 << 14];  
        rapidjson::FileReadStream frs(fp, buffer, sizeof(buffer));  
        rapidjson::Document d; d.ParseStream(frs);  
        fclose(fp);  
  
        auto getf=[&](const char* k, float &dst){ if (d.HasMember(k)) dst = d[k].GetFloat(); };  
        auto geti=[&](const char* k, int &dst){ if (d.HasMember(k)) dst = d[k].GetInt(); };  
        auto getb=[&](const char* k, bool &dst){ if (d.HasMember(k)) dst = d[k].GetBool(); };  
        auto gets=[&](const char* k, std::string &dst){ if (d.HasMember(k)) dst = d[k].GetString(); };  
  
        getf("det_bin_thresh", params->det_bin_thresh);  
        getf("det_box_thresh", params->det_box_thresh);  
        getf("det_unclip_ratio", params->det_unclip_ratio);  
        geti("det_max_candidates", params->det_max_candidates);  
        getf("det_min_box_size", params->det_min_box_size);  
        gets("det_output_name", params->det_output_name);  
        geti("det_map_h", params->det_map_h);  
        geti("det_map_w", params->det_map_w);  
        getb("letterbox_fix", params->letterbox_fix);  
  
        gets("rec_output_name", params->rec_output_name);  
        gets("charset_path", params->charset_path);  
        geti("blank_index", params->blank_index);  
        getb("logits_are_softmax", params->logits_are_softmax);  
        getb("time_major", params->time_major);  
        getf("text_conf_smooth", params->text_conf_smooth);  
        getb("attach_caption_box", params->attach_caption_box);  
    }  
  
    load_charset_from_file(*params);  
    return params;  
}  
  
void free_resources(void *params_void_ptr) {  
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
    delete p;  
}  
  
// ---------------------------  
// Tensor helpers (typed access)  
// ---------------------------  
static cv::Mat tensor_to_probmap_u8_as_float(HailoTensorPtr t, int H, int W) {  
    const uint8_t *u8 = reinterpret_cast<const uint8_t*>(t->data());  
    if (!u8) throw std::runtime_error("Detector tensor has null data()");  
    cv::Mat prob(H, W, CV_8UC1);  
    std::memcpy(prob.data, u8, (size_t)H * (size_t)W * sizeof(uint8_t));  
    cv::Mat out; prob.convertTo(out, CV_32F, 1.0/255.0);  
    return out;  
}  
  
static HailoTensorPtr get_tensor_by_name_or_fallback(const HailoROIPtr &roi, const std::string &desired) {  
    HailoTensorPtr chosen;  
    for (auto &t : roi->get_tensors()) { if (t->name() == desired) { chosen = t; break; } }  
    if (!chosen) {  
        auto tensors = roi->get_tensors();  
        if (tensors.empty()) throw std::runtime_error("ROI has no tensors");  
        chosen = tensors.front();  
    }  
    return chosen;  
}  

  
// ---------------------------  
// Detector (DB-like) postprocess - Enhanced based on Python DBPostProcess  
// ---------------------------  
static float region_score(const cv::Mat &prob, const std::vector<cv::Point> &poly) {  
    cv::Rect bbox = cv::boundingRect(poly) & cv::Rect(0,0,prob.cols,prob.rows);  
    if (bbox.empty()) return 0.f;  
    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8UC1);  
    std::vector<std::vector<cv::Point>> polys(1);  
    polys[0].reserve(poly.size());  
    for (auto &p : poly) polys[0].push_back(cv::Point(p.x - bbox.x, p.y - bbox.y));  
    cv::fillPoly(mask, polys, cv::Scalar(255));  
    cv::Scalar s = cv::mean(prob(bbox), mask);  
    return (float)s[0];  
}  
  
// Unclip polygon using offset (similar to Python's pyclipper)  
// static std::vector<cv::Point> unclip_polygon(const std::vector<cv::Point>& polygon, float unclip_ratio) {
//     // Convert input points to Clipper2 Path64
//     Path64 subj;
//     for (const auto &pt : polygon) {
//         subj.emplace_back(pt.x * 10, pt.y * 10);  // scale for integer precision
//     }

//     // Calculate offset distance
//     double area = Area(subj);
//     double perimeter = Perimeter(subj);
//     if (perimeter == 0) return polygon;
//     double offset_val = std::abs(area) * unclip_ratio / perimeter;

//     // Apply offset
//     ClipperOffset co;
//     co.AddPath(subj, JoinType::Round, EndType::Polygon);
//     Paths64 solution;
//     co.Execute(offset_val * 10, solution);  // match input scale

//     // Return first resulting path as cv::Points
//     std::vector<cv::Point> result;
//     if (!solution.empty()) {
//         for (const auto &pt : solution[0]) {
//             result.emplace_back(pt.x / 10, pt.y / 10);  // back to original scale
//         }
//     }
//     return result;
// }




// extern "C"  
// void paddleocr_det(HailoROIPtr roi, void *params_void_ptr) {  
//     if (!roi->has_tensors()) {  
//         std::cout << "DEBUG: paddleocr_det - No tensors in ROI" << std::endl;  
//         return;  
//     }  
//     auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
//     std::cout << "DEBUG: paddleocr_det called" << std::endl;  
      
//     std::cout << "DEBUG: Using thresholds - bin:" << p->det_bin_thresh   
//               << " box:" << p->det_box_thresh   
//               << " min_size:" << p->det_min_box_size << std::endl;  
  
//     // 1) fetch prob-map (UINT8 -> float [0..1])  
//     HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->det_output_name);  
//     const int H = p->det_map_h;  
//     const int W = p->det_map_w;  
//     cv::Mat prob = tensor_to_probmap_u8_as_float(t, H, W);  
  
//     // 2) threshold to binary  
//     cv::Mat bin;   
//     cv::threshold(prob, bin, p->det_bin_thresh, 1.0, cv::THRESH_BINARY);  
//     bin.convertTo(bin, CV_8U, 255.0);  
  
//     // 3) contours  
//     std::vector<std::vector<cv::Point>> contours;  
//     cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);  
//     std::cout << "DEBUG: Found " << contours.size() << " contours" << std::endl;  
  
//     // 4) build detections (bbox only) - BYPASS MODE  
//     std::vector<HailoDetection> outs;  
//     outs.reserve(std::min((int)contours.size(), p->det_max_candidates));  
  
//     // ROI letterbox region in original frame  
//     HailoBBox roi_box = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());  
//     const float sx = roi_box.width()  / (float)W;  
//     const float sy = roi_box.height() / (float)H;  
  
//     int pushed = 0;  
//     int contour_count = 0;  
//     for (auto &c : contours) {  
//         contour_count++;  
//         std::cout << "DEBUG: Processing contour " << contour_count << " with " << c.size() << " points" << std::endl;  
          
//         // More lenient point count filter - allow even single points for now  
//         if ((int)c.size() < 1) {  
//             std::cout << "DEBUG: Contour " << contour_count << " rejected - no points" << std::endl;  
//             continue;  
//         }  
  
//         // For single points or lines, use bounding rect instead of minAreaRect  
//         cv::Rect contour_bb = cv::boundingRect(c);  // Renamed from 'bb' to 'contour_bb'  
//         float short_side = std::min(contour_bb.width, contour_bb.height);  
//         std::cout << "DEBUG: Contour " << contour_count << " short_side: " << short_side << std::endl;  
          
//         // Very permissive size filter - even allow zero-size for testing  
//         if (short_side < 0) {  // Only reject negative sizes  
//             std::cout << "DEBUG: Contour " << contour_count << " rejected - invalid size" << std::endl;  
//             continue;  
//         }  
  
//         float score = region_score(prob, c);  
//         std::cout << "DEBUG: Contour " << contour_count << " score: " << score << std::endl;  
          
//         // Keep your permissive score threshold  
//         if (score < p->det_box_thresh) {  
//             std::cout << "DEBUG: Contour " << contour_count << " rejected - low score" << std::endl;  
//             continue;  
//         }  
  
//         std::cout << "DEBUG: Contour " << contour_count << " ACCEPTED - creating detection" << std::endl;  
  
//         // Unclip the polygon (expand it)  
//         // std::vector<cv::Point> unclipped = unclip_polygon(c, p->det_unclip_ratio);  
//         std::vector<cv::Point> unclipped = ocrpp_unclip_polygon(c, p->det_unclip_ratio);

          
//         // Use bounding rect of unclipped polygon  
//         cv::Rect final_bb = cv::boundingRect(unclipped);  // Renamed from 'bb' to 'final_bb'  
        
//         // Additional quality filters
//         float aspect_ratio = (float)final_bb.width / final_bb.height;
//         float area_px = final_bb.width * final_bb.height;

//         // Filter by aspect ratio (text is usually wider than tall)
//         if (aspect_ratio < 0.3f || aspect_ratio > 10.0f) {
//             std::cout << "DEBUG: Contour " << contour_count << " rejected - bad aspect ratio: " << aspect_ratio << std::endl;
//             continue;
//         }

//         // Filter by minimum area in pixels 
//         // if (area_px < 100.0f) {  // Adjust threshold as needed
//         //     std::cout << "DEBUG: Contour " << contour_count << " rejected - too small area: " << area_px << std::endl;
//         //     continue;
//         // }

    
//         // Map to frame coords (letterbox fix)  
//         float xmin = final_bb.x * sx + roi_box.xmin();  
//         float ymin = final_bb.y * sy + roi_box.ymin();  
//         float w    = final_bb.width  * sx;  
//         float h    = final_bb.height * sy;  
  
//         outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);  
//         if (++pushed >= p->det_max_candidates) break;  
//     }  
  
//     std::cout << "DEBUG: Created " << outs.size() << " text region detections from " << contours.size() << " contours" << std::endl;  
  
//     if (!outs.empty()) {  
//         hailo_common::add_detections(roi, outs);  
//         if (p->letterbox_fix) roi->clear_scaling_bbox();  
//     }  
// }

//---------------------------  
// Recognizer (CTC greedy) - Enhanced based on Python implementation  
// ---------------------------  
static void softmax1d(std::vector<float> &v) {  
    float m = *std::max_element(v.begin(), v.end());  
    double sum = 0.0;  
    for (float &x : v) sum += std::exp(double(x - m));  
    for (float &x : v) x = float(std::exp(double(x - m)) / sum);  
}  
  
extern "C"  
void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr) {  
    if (!roi->has_tensors()) {  
        std::cout << "DEBUG: paddleocr_recognize - No tensors in ROI" << std::endl;  
        return;  
    }  
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
    std::cout << "DEBUG: paddleocr_recognize called" << std::endl;  
  
    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->rec_output_name);  
    const auto &shape = t->shape(); // FCR(1x40x97) => size()==3  
    if (shape.size() != 3) throw std::runtime_error("Unexpected recognizer rank (expected 3)");  
  
    // Pull UINT8 -> float probs in [0..1]  
    const uint8_t *u8 = reinterpret_cast<const uint8_t*>(t->data());  
    if (!u8) throw std::runtime_error("Recognizer tensor not UINT8");  
    const size_t N = shape[0];                  // 1  
    const size_t D1 = shape[1];                 // 40 or 97  
    const size_t D2 = shape[2];                 // 97 or 40  
    if (N != 1) throw std::runtime_error("Recognizer expects N=1");  
  
    // Heuristic: treat larger of (D1,D2) as T (timesteps), smaller as C (classes)  
    size_t C = std::min(D1, D2);  
    size_t T = std::max(D1, D2);  
    bool layout_is_NCT = (D1 == C && D2 == T);  // [N, C, T]  
      
    std::cout << "DEBUG: Tensor shape - N:" << N << " D1:" << D1 << " D2:" << D2 << " C:" << C << " T:" << T << std::endl;  
      
    // Build probs[T][C]  
    std::vector<std::vector<float>> probs(T, std::vector<float>(C));  
  
    // copy & normalize  
    const uint8_t *base = u8; // N=1, contiguous  
    if (layout_is_NCT) {  
        // [1, C, T]  
        for (size_t c=0; c<C; ++c) {  
            for (size_t t0=0; t0<T; ++t0) {  
                float v = base[c*T + t0] * (1.0f/255.0f);  
                probs[t0][c] = p->logits_are_softmax ? v : v; // if not softmax, will softmax below  
            }  
        }  
    } else {  
        // [1, T, C]  
        for (size_t t0=0; t0<T; ++t0) {  
            for (size_t c=0; c<C; ++c) {  
                float v = base[t0*C + c] * (1.0f/255.0f);  
                probs[t0][c] = p->logits_are_softmax ? v : v;  
            }  
        }  
    }  
  
    if (!p->logits_are_softmax) {  
        for (size_t t0=0; t0<T; ++t0) softmax1d(probs[t0]);  
    }  
  
    // Greedy CTC decode - similar to Python ocr_eval_postprocess  
    std::string out_text;  
    out_text.reserve(T);  
    float conf_sum = 0.f;   
    int conf_count = 0;  
    int prev = -1;  
      
    for (size_t t0=0; t0<T; ++t0) {  
        auto &row = probs[t0];  
        auto it = std::max_element(row.begin(), row.end());  
        int idx = int(std::distance(row.begin(), it));  
        float pmax = *it;  
          
        // CTC decoding: skip blank and repeated characters  
        if (idx != p->blank_index && idx != prev) {  
            if (idx >= 0 && idx < (int)p->charset.size()) {  
                out_text += p->charset[idx];  
            } else {  
                out_text += "?";  
            }  
            conf_sum += pmax;  
            conf_count++;  
        }  
        prev = idx;  
    }  
      
    float conf = (conf_count > 0) ? (conf_sum / (float)conf_count) : 0.f;  
      
    std::cout << "DEBUG: Decoded text: '" << out_text << "' confidence: " << conf << std::endl;  
  
    // Create classification objects and attach to existing detections  
    if (!out_text.empty() && out_text != " ") {  
        // Get existing detections from the ROI  
        auto detections = hailo_common::get_hailo_detections(roi);  
        std::cout << "DEBUG: Found " << detections.size() << " detections to attach classification to" << std::endl;  
          
        if (!detections.empty()) {  
            // Add classification to the first detection  
            auto classification = std::make_shared<HailoClassification>("license_plate", out_text, conf);  
            detections[0]->add_object(classification);  
            std::cout << "DEBUG: Added classification '" << out_text << "' to detection" << std::endl;  
        } else {  
            std::cout << "DEBUG: No detections found to attach classification to" << std::endl;  
        }  
    } else {  
        std::cout << "DEBUG: Empty or whitespace-only text, not creating classification" << std::endl;  
    }  
}  
  
extern "C"  
void crop_text_regions_filter(HailoROIPtr roi, void *params_void_ptr) {  
    std::cout << "DEBUG: crop_text_regions_filter called as hailofilter" << std::endl;  
      
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);  
    std::cout << "DEBUG: Found " << detections.size() << " detections in ROI" << std::endl;  
      
    std::vector<HailoDetectionPtr> text_detections;  
      
    for (auto detection : detections) {  
        std::string label = detection->get_label();  
        std::cout << "DEBUG: Processing detection with label: '" << label << "'" << std::endl;  
          
        // // Only keep text_region detections, filter out others  
        // if (label == "text_region") {  
        //     std::cout << "DEBUG: Found text_region detection - keeping for OCR" << std::endl;  
        //     text_detections.push_back(detection);  
        // } else {  
        //     std::cout << "DEBUG: Skipping detection with label: '" << label << "'" << std::endl;  
        // }  
        text_detections.push_back(detection); 
    }  
      
    // Remove all detections first  
    roi->remove_objects_typed(HAILO_DETECTION);  
      
    // Add back only text region detections  
    for (auto text_detection : text_detections) {  
        roi->add_object(text_detection);  
    }  
      
    std::cout << "DEBUG: Filtered to " << text_detections.size() << " text region detections" << std::endl;  
}

extern "C"
std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image,
                                           HailoROIPtr roi,
                                           bool use_letterbox,
                                           bool no_scaling_bbox,
                                           bool internal_offset,
                                           const std::string &resize_method)
{
    std::cout << "DEBUG: crop_text_regions called as cropper function" << std::endl;

    std::vector<HailoROIPtr> out_rois;
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    if (!image || !roi) {
        std::cout << "DEBUG: crop_text_regions - missing image/roi" << std::endl;
        return out_rois; // empty
    }

    const int img_w = image->width();
    const int img_h = image->height();

    constexpr int   MAX_TEXT_REGIONS  = 8;
    constexpr float MIN_W_PX          = 4.0f;
    constexpr float MIN_H_PX          = 2.0f;
    constexpr float TARGET_MIN_H_PX   = 12.0f;
    constexpr float PAD_X_PX          = 4.0f;
    constexpr float PAD_Y_PX          = 2.0f;

    auto clamp01 = [](float v){ return std::max(0.0f, std::min(1.0f, v)); };

    int count = 0;
    for (auto &detection : detections) {
        if (!detection) continue;
        if (count >= MAX_TEXT_REGIONS) break;

        if (detection->get_label() != "text_region") continue;

        HailoBBox nb = detection->get_bbox(); // normalized [0,1]
        float nx = nb.xmin();
        float ny = nb.ymin();
        float nw = nb.width();
        float nh = nb.height();

        if (use_letterbox) {
            float img_aspect = (float)img_w / (float)img_h;
            float scale = 1.0f, pad_x = 0.0f, pad_y = 0.0f;

            if (img_aspect >= 1.0f) {          // wide image
                scale = 1.0f / img_aspect;
                pad_x = (1.0f - scale) * 0.5f;
            } else {                            // tall image
                scale = img_aspect;
                pad_y = (1.0f - scale) * 0.5f;
            }

            float x0 = clamp01((nx      - pad_x) / scale);
            float y0 = clamp01((ny      - pad_y) / scale);
            float x1 = clamp01((nx + nw - pad_x) / scale);
            float y1 = clamp01((ny + nh - pad_y) / scale);

            nx = x0; ny = y0; nw = std::max(0.0f, x1 - x0); nh = std::max(0.0f, y1 - y0);
        }

        float w_px = nw * img_w;
        float h_px = nh * img_h;

        if (w_px < MIN_W_PX || h_px < MIN_H_PX) {
            std::cout << "DEBUG: Skipping text_region (too small in px)\n";
            continue;
        }

        if (h_px < TARGET_MIN_H_PX) {
            float center_y = ny + nh * 0.5f;
            float new_h_n  = TARGET_MIN_H_PX / (float)img_h;
            float new_y    = center_y - new_h_n * 0.5f;

            ny = clamp01(new_y);
            nh = std::min(1.0f - ny, new_h_n);
            h_px = nh * img_h;
        }

        const float pad_x_n = PAD_X_PX / (float)img_w;
        const float pad_y_n = PAD_Y_PX / (float)img_h;

        float x0 = clamp01(nx - pad_x_n);
        float y0 = clamp01(ny - pad_y_n);
        float x1 = clamp01(nx + nw + pad_x_n);
        float y1 = clamp01(ny + nh + pad_y_n);

        detection->set_bbox(HailoBBox(x0, y0,
                                      std::max(0.0f, x1 - x0),
                                      std::max(0.0f, y1 - y0)));
        ++count;
    }

    // Return the parent ROI (with updated detections).
    if (count > 0) {
        if (!no_scaling_bbox) {
            // keep parent's scaling bbox as-is (no changes)
        }
        out_rois.push_back(roi);
    }

    std::cout << "DEBUG: Returning " << out_rois.size() << " ROI(s) for cropping" << std::endl;
    return out_rois;
}


// --- helpers ---------------------------------------------------------------
static inline int odd_at_least(int v) { return (v % 2 == 0) ? v + 1 : v; }

static void morph_close_horizontal(cv::Mat &bin) {
// Heuristic kernel relative to prob-map size;
// wider than tall to join characters into words/plates.
const int W = bin.cols, H = bin.rows;
const int kx = odd_at_least(std::max(3, (int)std::round(W * 0.015))); // ~1.5% of width
const int ky = odd_at_least(std::max(1, (int)std::round(H * 0.008))); // ~0.8% of height
cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kx, ky));
cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k, {-1,-1}, 1);
}

static void merge_horizontal_boxes(std::vector<cv::Rect> &rects,
                                   int max_gap_px,
                                   float min_y_overlap_ratio)
{
    if (rects.size() <= 1) return;

    std::sort(rects.begin(), rects.end(),
              [](const cv::Rect &a, const cv::Rect &b){ return a.x < b.x; });

    auto y_overlap_ratio = [](const cv::Rect &a, const cv::Rect &b){
        const int top = std::max(a.y, b.y);
        const int bot = std::min(a.y + a.height, b.y + b.height);
        const int inter = std::max(0, bot - top);
        const int min_h = std::max(1, std::min(a.height, b.height));
        return static_cast<float>(inter) / static_cast<float>(min_h);
    };

    std::vector<cv::Rect> merged;
    merged.reserve(rects.size());
    cv::Rect run = rects[0];

    for (size_t i = 1; i < rects.size(); ++i) {
        const cv::Rect &next = rects[i];
        const int gap = next.x - (run.x + run.width);
        const float yov = y_overlap_ratio(run, next);
        if (gap <= max_gap_px && yov >= min_y_overlap_ratio) {
            run |= next; // union
        } else {
            merged.push_back(run);
            run = next;
        }
    }
    merged.push_back(run);
    rects.swap(merged);
}


static float region_score_rect(const cv::Mat &prob, const cv::Rect &r) {
const cv::Rect R = r & cv::Rect(0,0,prob.cols, prob.rows);
if (R.empty()) return 0.f;
return (float)cv::mean(prob(R))[0];
}

// DB-style "unclip" for rectangles, done iteratively with small steps.
// Each iteration grows the rect by d = (area * ratio) / perimeter, then clamps.
static cv::Rect db_unclip_rect_iter(cv::Rect r,
                                    float ratio_step,
                                    int iters,
                                    int W, int H,
                                    float max_grow_frac = 0.20f)  // hard cap, e.g. 20% of max dimension
{
    if (r.width <= 0 || r.height <= 0) return r;
    const int MAX_GROW = std::max(1, int(std::round(std::max(W, H) * max_grow_frac)));

    auto clamp_rect = [&](cv::Rect q){
        q.x = std::max(0, std::min(q.x, W-1));
        q.y = std::max(0, std::min(q.y, H-1));
        q.width  = std::max(1, std::min(q.width,  W - q.x));
        q.height = std::max(1, std::min(q.height, H - q.y));
        return q;
    };

    int total_grow_x = 0, total_grow_y = 0;
    for (int i = 0; i < iters; ++i) {
        const double A = double(r.width) * double(r.height);
        const double P = double(r.width + r.height) * 2.0;
        int d = int(std::round((A / std::max(1.0, P)) * ratio_step));
        d = std::max(1, d);  // at least 1px

        // don't exceed the cap
        if (total_grow_x + d > MAX_GROW || total_grow_y + d > MAX_GROW) break;

        cv::Rect nr(r.x - d, r.y - d, r.width + 2*d, r.height + 2*d);
        nr = clamp_rect(nr);

        if (nr == r) break; // no change -> stop
        r = nr;
        total_grow_x += d;
        total_grow_y += d;
    }
    return r;
}

static cv::Rect db_unclip_rect_iter_aniso(cv::Rect r,
                                          float ratio_x, float ratio_y,
                                          int iters, int W, int H,
                                          float max_grow_frac_x = 0.15f,
                                          float max_grow_frac_y = 0.08f)
{
    if (r.width <= 0 || r.height <= 0) return r;

    auto clamp = [&](cv::Rect q){
        q.x = std::max(0, std::min(q.x, W-1));
        q.y = std::max(0, std::min(q.y, H-1));
        q.width  = std::max(1, std::min(q.width,  W - q.x));
        q.height = std::max(1, std::min(q.height, H - q.y));
        return q;
    };

    const int max_gx = std::max(1, int(std::round(W * max_grow_frac_x)));
    const int max_gy = std::max(1, int(std::round(H * max_grow_frac_y)));
    int accx = 0, accy = 0;

    for (int i = 0; i < iters; ++i) {
        const double A = double(r.width) * double(r.height);
        const double P = double(r.width + r.height) * 2.0;
        const double base = A / std::max(1.0, P);

        int dx = std::max(1, int(std::round(base * ratio_x)));
        int dy = std::max(1, int(std::round(base * ratio_y)));

        // cap growth
        dx = std::min(dx, std::max(0, max_gx - accx));
        dy = std::min(dy, std::max(0, max_gy - accy));
        if (dx == 0 && dy == 0) break;

        r = clamp(cv::Rect(r.x - dx, r.y - dy, r.width + 2*dx, r.height + 2*dy));
        accx += dx; accy += dy;
    }
    return r;
}



extern "C"
void paddleocr_det(HailoROIPtr roi, void *params_void_ptr) {
    if (!roi->has_tensors()) {
        std::cout << "DEBUG: paddleocr_det - No tensors in ROI\n";
        return;
    }
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);
    std::cout << "DEBUG: paddleocr_det called\n";
    std::cout << "DEBUG: Using thresholds - bin:" << p->det_bin_thresh
              << " box:" << p->det_box_thresh
              << " min_size:" << p->det_min_box_size << " max_candidates:" << p->det_max_candidates << std::endl;

    // ---- 1) Fetch tensor & robust H,W --------------------------------------
    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->det_output_name);
    const auto &sh = t->shape();
    std::cout << "DEBUG: det tensor full shape: [";
    for (size_t i=0;i<sh.size();++i) std::cout<<sh[i]<<(i+1<sh.size()? ",":"");
    std::cout<<"]\n";

    int H=0,W=0;
    if (sh.size()==4) {
        if (sh[1]==1)      { H=int(sh[2]); W=int(sh[3]); } // NCHW
        else if (sh[3]==1) { H=int(sh[1]); W=int(sh[2]); } // NHWC
        else               { H=int(sh[2]); W=int(sh[3]); }
    } else if (sh.size()==3) {
        if (sh[2]==1)      { H=int(sh[0]); W=int(sh[1]); } // [H,W,1]
        else if (sh[0]==1) { H=int(sh[1]); W=int(sh[2]); } // [1,H,W]
        else {
            std::vector<int> v{int(sh[0]),int(sh[1]),int(sh[2])};
            std::sort(v.begin(), v.end());
            H=v[1]; W=v[2];
        }
    } else if (sh.size()==2) { H=int(sh[0]); W=int(sh[1]); }
    else { H=p->det_map_h; W=p->det_map_w; }
    if (W<=4 && H>16) std::swap(H,W);
    std::cout << "DEBUG: resolved H="<<H<<" W="<<W << std::endl;

    // ---- 2) prob map + stats ----------------------------------------------
    cv::Mat prob = tensor_to_probmap_u8_as_float(t, H, W);

    cv::imwrite("debug_prob.png", prob * 255); 
    double mn,mx; cv::minMaxLoc(prob, &mn, &mx);
    cv::Scalar meanv = cv::mean(prob);
    int above_default = cv::countNonZero(prob > p->det_bin_thresh);
    float fg_ratio_default = float(above_default) / float(H*W);
    std::cout << "DEBUG: prob stats min="<<mn<<" max="<<mx<<" mean="<<meanv[0]
              << " above@bin="<<above_default<<"/"<<(H*W)
              << " ("<<fg_ratio_default<<")\n";

    // ---- 3) adaptive threshold + closing ----------------------------------
    float bin_thr = p->det_bin_thresh;
    if      (fg_ratio_default < 0.003f) bin_thr = std::max(0.15f, p->det_bin_thresh * 0.8f);  // sparse -> be lenient
    else if (fg_ratio_default > 0.08f)  bin_thr = std::min(0.75f, p->det_bin_thresh * 1.2f);  // dense  -> be stricter
    std::cout << "DEBUG: chosen bin_thr="<<bin_thr<<" (from "<<p->det_bin_thresh<<")\n";

    cv::Mat bin;
    cv::threshold(prob, bin, bin_thr, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255.0);

    cv::imwrite("debug_binary.png", bin);  

    // closing kernel scales with density; keep moderate to avoid 1 mega-blob
    float kscale = (fg_ratio_default < 0.01f) ? 1.0f : (fg_ratio_default > 0.06f ? 1.5f : 1.2f);
    int kx = odd_at_least(std::max(3, int(std::round(W * 0.012f * kscale))));
    int ky = odd_at_least(std::max(1, int(std::round(H * 0.006f * kscale))));
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kx, ky));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k, {-1,-1}, 1);
    std::cout << "DEBUG: morph close kernel kx="<<kx<<" ky="<<ky<<"\n";

    // ---- 4) contours -> rects ---------------------------------------------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::cout << "DEBUG: contours="<<contours.size()<<" (post-close)\n";

    std::vector<cv::Rect> rects; rects.reserve(contours.size());
    for (auto &c : contours) {
        if (c.empty()) continue;
        cv::Rect r = cv::boundingRect(c);
        if (r.width>0 && r.height>0) rects.push_back(r);
    }
    std::cout << "DEBUG: initial rects="<<rects.size()<<std::endl;

    if (rects.empty()) {
        std::cout<<"DEBUG: no rects -> exit early\n";
        return;
    }

    // ---- 4.5) merge horizontally (careful not to over-merge) ---------------
    std::vector<int> hs; hs.reserve(rects.size());
    for (auto &r : rects) hs.push_back(r.height);
    std::nth_element(hs.begin(), hs.begin()+hs.size()/2, hs.end());
    const int median_h = hs[hs.size()/2];

    // smaller gap to keep multiple boxes; y overlap still needs to be decent
    int   GAP_PX = std::max(3, std::min(int(W * 0.02f), int(std::round(median_h * 1.0f))));
    float YOV    = 0.45f;
    merge_horizontal_boxes(rects, GAP_PX, YOV);
    std::cout << "DEBUG: merged rects="<<rects.size()<<" (gap="<<GAP_PX<<", yov="<<YOV<<", median_h="<<median_h<<")\n";

    // local clamp helper in function scope (prevents linker/visibility issues)
    auto clamp_rect_xywh = [&](cv::Rect r)->cv::Rect{
        r.x = std::max(0, std::min(r.x, W - 1));
        r.y = std::max(0, std::min(r.y, H - 1));
        r.width  = std::max(1, std::min(r.width,  W - r.x));
        r.height = std::max(1, std::min(r.height, H - r.y));
        return r;
    };

    // ---- 4.6) Grow (unclip-like) each rect on all sides --------------------
    // Base padding from median height; plus two small iterative grows
    const int   PAD_X0 = std::max(2, int(std::round(median_h * 0.6f)));
    const int   PAD_Y0 = std::max(1, int(std::round(median_h * 0.35f)));
    const int   GROW_ITERS = 2;
    const float GROW_X_PER_H = 0.15f;   // per-iter grow relative to current height
    const float GROW_Y_PER_H = 0.12f;
    std::cout << "DEBUG: inflate params PAD_X0="<<PAD_X0<<" PAD_Y0="<<PAD_Y0
              << " GROW_ITERS="<<GROW_ITERS
              << " GROW_X_PER_H="<<GROW_X_PER_H
              << " GROW_Y_PER_H="<<GROW_Y_PER_H << "\n";

    for (size_t i=0;i<rects.size();++i) {
        cv::Rect r = rects[i];
        float ar0 = float(r.width) / std::max(1, r.height);
        float score0 = region_score_rect(prob, r);
        int area0 = r.width * r.height;
        std::cout << "DEBUG: pre-inflate rect#"<<(i+1)<<" x="<<r.x<<" y="<<r.y
                  <<" w="<<r.width<<" h="<<r.height<<" ar="<<ar0
                  <<" area="<<area0<<" score="<<score0<<"\n";

        // base inflate
        r = clamp_rect_xywh(cv::Rect(r.x - PAD_X0, r.y - PAD_Y0,
                                     r.width + 2*PAD_X0, r.height + 2*PAD_Y0));

        // extra vertical thickening for very wide lines
        float ar_after_base = float(r.width) / std::max(1, r.height);
        if (ar_after_base > 10.0f) {
            int addY = std::max(PAD_Y0, int(std::round(r.height * 0.5f)));
            r = clamp_rect_xywh(cv::Rect(r.x, r.y - addY/2,
                                         r.width, r.height + addY));
        }

        // couple of gentle iterative grows
        for (int it=0; it<GROW_ITERS; ++it) {
            int gx = std::max(1, int(std::round(std::max(2.0f, r.height * GROW_X_PER_H))));
            int gy = std::max(1, int(std::round(std::max(1.0f, r.height * GROW_Y_PER_H))));
            r = clamp_rect_xywh(cv::Rect(r.x - gx, r.y - gy,
                                         r.width + 2*gx, r.height + 2*gy));
        }

        float ar1 = float(r.width) / std::max(1, r.height);
        float score1 = region_score_rect(prob, r);
        int area1 = r.width * r.height;
        std::cout << "DEBUG: post-inflate rect#"<<(i+1)<<" x="<<r.x<<" y="<<r.y
                  <<" w="<<r.width<<" h="<<r.height<<" ar="<<ar1
                  <<" area="<<area1<<" score="<<score1<<"\n";

        rects[i] = r;
    }

    // ---- 5) Score/filter + map --------------------------------------------
    HailoBBox roi_box = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    const float sx = roi_box.width()  / float(W);
    const float sy = roi_box.height() / float(H);

    // loose filters to avoid dropping everything
    const int   MIN_H_PX     = std::max(3, int(std::round(H * 0.010f)));        // kill slivers
    const float AR_MIN       = 0.6f;
    const float AR_MAX       = 80.0f;
    const float MIN_AREA_PX  = std::max(80.0f, float(median_h * median_h * 0.4f));
    const float SCORE_BASE   = p->det_box_thresh;

    std::cout << "DEBUG: filters MIN_H_PX="<<MIN_H_PX<<" AR_MIN="<<AR_MIN<<" AR_MAX="<<AR_MAX
              <<" MIN_AREA_PX="<<MIN_AREA_PX<<" SCORE_BASE="<<SCORE_BASE<<"\n";

    std::vector<HailoDetection> outs; outs.reserve(rects.size());
    for (size_t i=0;i<rects.size();++i) {
        const auto &r = rects[i];
        float ar = float(r.width) / std::max(1, r.height);
        int   hpx = r.height;
        float area = float(r.width) * float(r.height);
        float score = region_score_rect(prob, r);
        float score_min = (ar > 16.0f) ? std::max(0.45f, SCORE_BASE - 0.15f) : SCORE_BASE;

        if (hpx < MIN_H_PX)           { std::cout<<"DEBUG: rect#"<<(i+1)<<" drop minH "<<hpx<<"\n"; continue; }
        if (area < MIN_AREA_PX)       { std::cout<<"DEBUG: rect#"<<(i+1)<<" drop area "<<area<<"\n"; continue; }
        if (ar < AR_MIN || ar > AR_MAX){ std::cout<<"DEBUG: rect#"<<(i+1)<<" drop AR="<<ar<<"\n"; continue; }
        if (score < score_min)        { std::cout<<"DEBUG: rect#"<<(i+1)<<" drop score="<<score<<" < "<<score_min<<"\n"; continue; }

        float xmin = r.x * sx + roi_box.xmin();
        float ymin = r.y * sy + roi_box.ymin();
        float w    = r.width  * sx;
        float h    = r.height * sy;

        outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);
        std::cout << "DEBUG: keep rect#"<<(i+1)<<" -> bbox{x="<<xmin<<",y="<<ymin<<",w="<<w<<",h="<<h<<"} score="<<score<<"\n";
        if ((int)outs.size() >= p->det_max_candidates) {
            std::cout << "DEBUG: reached det_max_candidates="<<p->det_max_candidates<<"\n";
            break;
        }
    }

    // ---- 5.5) Fallback: keep the 2 widest rects if we filtered everything --
    if (outs.empty() && !rects.empty()) {
        std::cout << "DEBUG: no survivors; FALLBACK keep up to 2 widest rects\n";
        std::vector<size_t> idx(rects.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            return rects[a].width > rects[b].width;
        });
        const size_t keepN = std::min<size_t>(2, idx.size());
        for (size_t k=0; k<keepN; ++k) {
            cv::Rect r = rects[idx[k]];
            // final tiny pad
            int gx = std::max(1, int(std::round(r.height * 0.1f)));
            int gy = std::max(1, int(std::round(r.height * 0.1f)));
            r = clamp_rect_xywh(cv::Rect(r.x - gx, r.y - gy, r.width + 2*gx, r.height + 2*gy));
            float xmin = r.x * sx + roi_box.xmin();
            float ymin = r.y * sy + roi_box.ymin();
            float w    = r.width  * sx;
            float h    = r.height * sy;
            float score = region_score_rect(prob, r);
            outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);
            std::cout << "DEBUG: FALLBACK keep rect idx="<<idx[k]<<" w="<<r.width<<" h="<<r.height<<" score="<<score<<"\n";
        }
    }

    std::cout << "DEBUG: Created "<<outs.size()<<" text detections\n";
    if (!outs.empty()) {
        hailo_common::add_detections(roi, outs);
        if (p->letterbox_fix) roi->clear_scaling_bbox();
    }
}
