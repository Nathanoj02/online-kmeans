#include "sequential_submodule.hpp"

#include "core.hpp"

#include <iostream>

namespace signals {

} // namespace signals


using namespace signals;


void sequential_submodule(pybind11::module& subm)
{
    subm.doc() = "Python signals submodule for sequential algorithms";

    subm.def("foo", &print);

    // subm.def(
    //     "census_transform", 
    //     [](py::array_t<std::uint8_t> img_arr, py::tuple kernel_shape){
    //         py::buffer_info img_buf = img_arr.request();

    //         if (img_buf.ndim != 2)
    //             throw std::runtime_error("Number of dimensions must be 2");

    //         auto result = py::array_t<std::uint64_t>(
    //             {img_buf.shape[0], img_buf.shape[1]});
    //         result[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info result_buf = result.request();
            
    //         census_transform(
    //             static_cast<std::uint64_t*>(result_buf.ptr), 
    //             static_cast<std::uint8_t*>(img_buf.ptr), 
    //             img_buf.shape[0], img_buf.shape[1], 
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>());

    //         return result;
    //     });
    
    // subm.def(
    //     "cost_volume", 
    //     [](py::array_t<std::uint8_t> left_img_arr, 
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

    //         auto left_census = py::array_t<std::uint64_t>(
    //             {left_img_buf.shape[0], left_img_buf.shape[1]});
    //         left_census[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info left_census_buf = left_census.request();
    //         auto right_census = py::array_t<std::uint64_t>(
    //             {right_img_buf.shape[0], right_img_buf.shape[1]});
    //         right_census[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info right_census_buf = right_census.request();

    //         auto result = py::array_t<std::uint8_t>(
    //             {left_img_buf.shape[0], left_img_buf.shape[1], py::ssize_t(max_disparity)});
    //         result[py::make_tuple(py::ellipsis())] = 0;
    //         py::buffer_info result_buf = result.request();

    //         census_transform(
    //             static_cast<std::uint64_t*>(left_census_buf.ptr), 
    //             static_cast<std::uint8_t*>(left_img_buf.ptr), 
    //             left_img_buf.shape[0], left_img_buf.shape[1], 
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>());
    //         census_transform(
    //             static_cast<std::uint64_t*>(right_census_buf.ptr), 
    //             static_cast<std::uint8_t*>(right_img_buf.ptr), 
    //             right_img_buf.shape[0], right_img_buf.shape[1], 
    //             kernel_shape[0].cast<std::size_t>(), kernel_shape[1].cast<std::size_t>());
            
    //         cost_volume(
    //             static_cast<std::uint8_t*>(result_buf.ptr),
    //             static_cast<std::uint64_t*>(left_census_buf.ptr),
    //             static_cast<std::uint64_t*>(right_census_buf.ptr),
    //             max_disparity, left_img_buf.shape[0], left_img_buf.shape[1]
    //         );

    //         return result;
    //     });

    // subm.def(
    //     "select_disparity", 
    //     [](py::array_t<std::int16_t> aggr_volume){
    //         py::buffer_info aggr_volume_buf = aggr_volume.request();

    //         if (aggr_volume_buf.ndim != 4)
    //             throw std::runtime_error("Number of dimensions must be 4");

    //         auto result = py::array_t<std::int16_t>(
    //             {aggr_volume_buf.shape[0], aggr_volume_buf.shape[1]});
    //         py::buffer_info result_buf = result.request();
            
    //         select_disparity(
    //             static_cast<std::int16_t*>(result_buf.ptr), 
    //             static_cast<std::int16_t*>(aggr_volume_buf.ptr), 
    //             aggr_volume_buf.shape[0], aggr_volume_buf.shape[1], 
    //             aggr_volume_buf.shape[2], aggr_volume_buf.shape[3]);

    //         return result;
    //     });
}