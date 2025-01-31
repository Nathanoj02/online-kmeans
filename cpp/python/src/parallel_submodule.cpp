#include "parallel_submodule.hpp"

#include "core_par.hpp"

namespace signals {

} // namespace signals


using namespace signals;


void parallel_submodule(pybind11::module& subm)
{
    subm.doc() = "Python signals submodule for parallel algorithms";

    py::class_<cuda::DeviceInfo>(subm, "DeviceInfo");
    py::class_<cuda::KmeansInfo>(subm, "KmeansInfo");

    subm.def(
        "init_k_means", 
        [](std::size_t img_height, std::size_t img_width, std::uint64_t k) {
            auto dev = cuda::init_k_means(img_height, img_width, k);
            return dev;
        });

    subm.def(
        "k_means",
        [](py::array_t<std::uint8_t> img_arr,
           std::uint64_t k, std::float_t stab_error,
           cuda::KmeansInfo& dev_info, bool use_shared_mem) {

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
                k, stab_error, dev_info, use_shared_mem
            );

            return result;
        });

    subm.def(
        "deinit_k_means", 
        [](cuda::KmeansInfo& dev_info) {
            cuda::deinit_k_means(dev_info);    
        });
}