#include "core_par.hpp"

#include "error.cuh"

#include <cuda.h>
#include <cuda_runtime.h> 


namespace signals {

namespace cuda {

// void k_means(
//     uint8_t* dst, uint8_t* img,
//     size_t img_height, size_t img_width,
//     uint64_t k, float_t stab_error)
// {
//     std::cout << "Ok" << std::endl;
// }

void print() {
    std::cout << "Ok" << std::endl;
}

static cudaDeviceProp find_best_gpu()
{
    // Save properties of GPUs
    int dev_count;
    SAFE_CALL( cudaGetDeviceCount(&dev_count) );

    // Save best GPU
    cudaDeviceProp dev_prop;
    cudaDeviceProp best_device_prop;

    for (int i = 0; i < dev_count; i++)
    {
        SAFE_CALL( cudaGetDeviceProperties(&dev_prop, i) );
        
        if (i == 0 
            || (dev_prop.maxThreadsPerMultiProcessor * dev_prop.multiProcessorCount > 
                best_device_prop.maxThreadsPerMultiProcessor * best_device_prop.multiProcessorCount))
        {
            best_device_prop = dev_prop;
        }
    }
    return best_device_prop;
}

DeviceInfo& find_best_grid(DeviceInfo& device_info, std::size_t height, std::size_t width)
{
    auto best_device = find_best_gpu();

    std::size_t threadsPerBlockTemp;

    if(height * width > best_device.maxThreadsPerBlock / 2)
        threadsPerBlockTemp = best_device.maxThreadsPerBlock;
    else
        threadsPerBlockTemp = pow(2, ceil(log2(height * width)));
    
    std::size_t threadsPerBlockRow = pow(2, ceil(log2(sqrt(threadsPerBlockTemp))));

    device_info.block = {
        threadsPerBlockRow, 
        threadsPerBlockRow, 
        1,
    };
    device_info.grid = {
        width / threadsPerBlockRow, 
        height / threadsPerBlockRow,
        1
    };
    if (width % threadsPerBlockRow) device_info.grid.x++;
    if (height % threadsPerBlockRow) device_info.grid.y++;
    return device_info;
}



} // namespace cuda

} // namespace signals