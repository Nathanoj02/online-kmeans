#include "parallel_submodule.hpp"

#include "core_par.hpp"

namespace signals {

} // namespace signals


using namespace signals;


void parallel_submodule(pybind11::module& subm)
{
    subm.doc() = "Python signals submodule for parallel algorithms";

    py::class_<cuda::DeviceInfo>(subm, "DeviceInfo");
    
    // subm.def(
    //     "k_means",
    //     [](py::array_t<std::uint8_t> img_arr,
    //        std::uint64_t k, std::float_t stab_error) {

    //         py::buffer_info img_buf = img_arr.request();

    //         if (img_buf.ndim != 3)
    //             throw std::runtime_error("Number of dimensions must be 3");
            
    //         auto result = py::array_t<std::uint8_t>(
    //             {img_buf.shape[0], img_buf.shape[1], img_buf.shape[2]});
    //         result[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info result_buf = result.request();

    //         k_means(
    //             static_cast<std::uint8_t*>(result_buf.ptr),
    //             static_cast<std::uint8_t*>(img_buf.ptr),
    //             img_buf.shape[0], img_buf.shape[1],
    //             k, stab_error
    //         );

    //         return result;
    //     });

    subm.def(
        "k_means",
        [](py::array_t<std::uint8_t> img_arr,
           std::uint64_t k, std::float_t stab_error) {

            py::buffer_info img_buf = img_arr.request();

            if (img_buf.ndim != 3)
                throw std::runtime_error("Number of dimensions must be 3");
            
            auto result = py::array_t<std::uint8_t>(
                {img_buf.shape[0], img_buf.shape[1], img_buf.shape[2]});
            result[py::make_tuple(py::ellipsis())] = 0;
            py::buffer_info result_buf = result.request();

            cuda::k_means(
                static_cast<std::uint8_t*>(result_buf.ptr),
                static_cast<std::uint8_t*>(img_buf.ptr),
                img_buf.shape[0], img_buf.shape[1],
                k, stab_error
            );

            return result;
        }
    );

    // py::class_<cuda::CensusTransformDeviceInfo>(subm, "CensusTransformDeviceInfo");
    // py::class_<cuda::CostVolumeDeviceInfo>(subm, "CostVolumeDeviceInfo");
    // py::class_<cuda::CostAggregationDeviceInfo>(subm, "CostAggregationDeviceInfo");
    // py::class_<cuda::SelectDisparityDeviceInfo>(subm, "SelectDisparityDeviceInfo");

    // subm.def(
    //     "init_census_transform", 
    //     [](std::size_t height, std::size_t width) {
    //         auto dev = cuda::init_census_transform(height, width);
    //         return dev;
    //     });

    // subm.def(
    //     "deinit_census_transform", 
    //     [](cuda::CensusTransformDeviceInfo& dev_info) {
    //         cuda::deinit_census_transform(dev_info);    
    //     });

    // subm.def(
    //     "census_transform", 
    //     [](const cuda::CensusTransformDeviceInfo& dev_info, py::array_t<std::uint8_t> img_arr, py::tuple kernel_shape){
    //         py::buffer_info img_buf = img_arr.request();

    //         if (img_buf.ndim != 2)
    //             throw std::runtime_error("Number of dimensions must be 2");

    //         auto result = py::array_t<std::uint64_t>(
    //             {img_buf.shape[0], img_buf.shape[1]});
    //         result[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info result_buf = result.request();
            
    //         cuda::census_transform(
    //             static_cast<std::uint64_t*>(result_buf.ptr), 
    //             static_cast<std::uint8_t*>(img_buf.ptr), 
    //             img_buf.shape[0], img_buf.shape[1], 
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>(), 
    //             dev_info);

    //         return result;
    //     });

    // subm.def(
    //     "init_cost_volume", 
    //     [](std::size_t height, std::size_t width, std::size_t max_disparity) {
    //         auto dev = cuda::init_cost_volume(height, width, max_disparity);
    //         return dev;
    //     });

    // subm.def(
    //     "deinit_cost_volume", 
    //     [](cuda::CostVolumeDeviceInfo& dev_info) {
    //         cuda::deinit_cost_volume(dev_info);    
    //     });
    
    // subm.def(
    //     "cost_volume", 
    //     [](cuda::CostVolumeDeviceInfo& dev_info, 
    //        py::array_t<std::uint8_t> left_img_arr, 
    //        py::array_t<std::uint8_t> right_img_arr, 
    //        std::size_t max_disparity,
    //        py::tuple kernel_shape){
    //         py::buffer_info left_img_buf = left_img_arr.request();
    //         py::buffer_info right_img_buf = right_img_arr.request();

    //         if (left_img_buf.ndim != 2)
    //             throw std::runtime_error("Number of dimensions must be 2");

    //         if (right_img_buf.ndim != 2)
    //             throw std::runtime_error("Number of dimensions must be 2");

    //         if (left_img_buf.shape[0] != right_img_buf.shape[0] 
    //             || left_img_buf.shape[1] != right_img_buf.shape[1])
    //             throw std::runtime_error("Dimensions must match");

    //         auto result = py::array_t<std::uint8_t>(
    //             {left_img_buf.shape[0], left_img_buf.shape[1], py::ssize_t(max_disparity)});
    //         result[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info result_buf = result.request();

    //         cuda::census_transform_input(
    //             static_cast<std::uint8_t*>(left_img_buf.ptr),
    //             left_img_buf.shape[0], left_img_buf.shape[1], dev_info.i_left
    //         );
    //         cuda::census_transform_input(
    //             static_cast<std::uint8_t*>(right_img_buf.ptr),
    //             right_img_buf.shape[0], right_img_buf.shape[1], dev_info.i_right
    //         );
    //         cuda::census_transform_exec(
    //             left_img_buf.shape[0], left_img_buf.shape[1],
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>(), dev_info.i_left
    //         );
    //         cuda::census_transform_exec(
    //             right_img_buf.shape[0], right_img_buf.shape[1],
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>(), dev_info.i_right
    //         );
    //         cuda::cost_volume_exec(
    //             left_img_buf.shape[0], left_img_buf.shape[1], max_disparity, dev_info
    //         );
    //         cuda::cost_volume_output(
    //             static_cast<std::uint8_t*>(result_buf.ptr),
    //             left_img_buf.shape[0], left_img_buf.shape[1], max_disparity, dev_info
    //         );

    //         return result;
    //     });

    // subm.def(
    //     "init_select_disparity", 
    //     [](std::size_t height, std::size_t width, std::size_t max_disparity, std::size_t path_num,
    //        std::uint8_t P1, std::uint8_t P2, std::size_t kp_num) {
    //         auto dev = cuda::init_select_disparity(
    //             height, width, max_disparity, path_num, P1, P2, kp_num);
    //         return dev;
    //     });

    // subm.def(
    //     "deinit_select_disparity", 
    //     [](cuda::SelectDisparityDeviceInfo& dev_info) {
    //         cuda::deinit_select_disparity(dev_info);    
    //     });

    // subm.def(
    //     "select_disparity", 
    //     [](cuda::SelectDisparityDeviceInfo& dev_info, py::array_t<std::int16_t> aggr_volume){
    //         py::buffer_info aggr_volume_buf = aggr_volume.request();

    //         if (aggr_volume_buf.ndim != 4)
    //             throw std::runtime_error("Number of dimensions must be 4");

    //         auto result = py::array_t<std::int16_t>(
    //             {aggr_volume_buf.shape[0], aggr_volume_buf.shape[1]});
    //         py::buffer_info result_buf = result.request();
            
    //         cuda::select_disparity(
    //             static_cast<std::int16_t*>(result_buf.ptr), 
    //             static_cast<std::int16_t*>(aggr_volume_buf.ptr), 
    //             aggr_volume_buf.shape[0], aggr_volume_buf.shape[1], 
    //             aggr_volume_buf.shape[2], aggr_volume_buf.shape[3],
    //             dev_info);

    //         return result;
    //     });
}