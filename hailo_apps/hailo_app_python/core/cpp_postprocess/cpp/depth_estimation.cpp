/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "depth_estimation.hpp"
#include "common/tensors.hpp"
#include <opencv2/opencv.hpp>          // for cv::Mat, applyColorMap
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>

static const char *output_layer_name_scdepth = "scdepthv3/conv31";
static const char *OUTPUT_LAYER_NAME        = "depth_anything_v2_small/conv79";

// Central dispatch: use Depth-Anything-V2 postprocess
void filter(HailoROIPtr roi)
{
    filter_depth_anything(roi);
}

// Existing ScDepth postprocess (unchanged)
void filter_scdepth(HailoROIPtr roi)
{
    if (!roi->has_tensors()) {
        return;
    }
    HailoTensorPtr tensor = roi->get_tensor(output_layer_name_scdepth);

    // 1) Dequantize
    xt::xarray<uint16_t> data_u16 = common::get_xtensor_uint16(tensor);
    xt::xarray<float> depth_rel    = common::dequantize(
        data_u16,
        tensor->vstream_info().quant_info.qp_scale,
        tensor->vstream_info().quant_info.qp_zp
    );

    // 2) Apply logistic-based scaling
    std::vector<float> tmp(depth_rel.begin(), depth_rel.end());
    xt::xarray<float> input = xt::adapt(tmp, {tensor->height(), tensor->width()});
    xt::xarray<float> output = xt::exp(-input);
    output = 1 / (1 + output);
    output = 1 / (output * 10 + 0.009);

    // 3) Wrap as float depth mask
    std::vector<float> out_vec(output.begin(), output.end());
    hailo_common::add_object(
        roi,
        std::make_shared<HailoDepthMask>(std::move(out_vec), tensor->width(), tensor->height(), 1.0f)
    );
}

// Depth-Anything-V2 postprocess with colour mapping (normalized RGB floats)
void filter_depth_anything(HailoROIPtr roi)
{
    if (!roi->has_tensors()) {
        return;
    }

    HailoTensorPtr tensor = roi->get_tensor(OUTPUT_LAYER_NAME);
    // 1) Dequantize to float relative depth
    xt::xarray<uint16_t> data_u16 = common::get_xtensor_uint16(tensor);
    xt::xarray<float> depth_rel   = common::dequantize(
        data_u16,
        tensor->vstream_info().quant_info.qp_scale,
        tensor->vstream_info().quant_info.qp_zp
    );

    // 2) Min–max normalize to [0,1]
    float dmin = xt::amin(depth_rel)();
    float dmax = xt::amax(depth_rel)();
    if (dmax > dmin) {
        depth_rel = (depth_rel - dmin) / (dmax - dmin);
    } else {
        depth_rel = xt::zeros_like(depth_rel);
    }

    // 3) Scale to [0,255] and cast to uint8
    xt::xarray<uint8_t> depth_u8 = xt::cast<uint8_t>(depth_rel * 255.0f);

    // 4) Create grayscale OpenCV Mat (0–255)
    int H = tensor->height();
    int W = tensor->width();
    cv::Mat gray_mat(H, W, CV_8UC1, depth_u8.data());

    // 5) Apply PLASMA colormap for colourful output
    cv::Mat color_mat;
    cv::applyColorMap(gray_mat, color_mat, cv::COLORMAP_PLASMA);

    // 6) Flatten into normalized RGB float vector [R,G,B,...] in [0,1]
    std::vector<float> rgb_vec;
    rgb_vec.reserve(H * W * 3);
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row = color_mat.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            // BGR order -> R,G,B normalized
            rgb_vec.push_back(row[x][2] / 255.0f);
            rgb_vec.push_back(row[x][1] / 255.0f);
            rgb_vec.push_back(row[x][0] / 255.0f);
        }
    }

    // 7) Add coloured depth mask (values already normalized)
    hailo_common::add_object(
        roi,
        std::make_shared<HailoDepthMask>(std::move(rgb_vec), W, H, 1.0f)
    );
}