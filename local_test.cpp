
#include <cuda.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize the CUDA Driver API
    CUresult result = cuInit(0);

    if (result != CUDA_SUCCESS) {
        std::cerr << "CUDA initialization failed\n";
        return 1;
    }

    // Select GPU 0
    CUdevice device;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::cerr << "GPU not found\n";
        return 1;
    }

    // Create a CUDA context
    CUcontext context;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::cerr << "Context creation failed\n";
        return 1;
    }

    // Load our compiler-generated GPU binary
    CUmodule module;
    if (cuModuleLoad(&module, "p00_generated.cubin") != CUDA_SUCCESS) {
        std::cerr << "Failed to load GPU binary\n";
        return 1;
    }

    // Find our PTX kernel entry point
    CUfunction kernel;
    if (cuModuleGetFunction(&kernel, module, "k_add_pointwise") != CUDA_SUCCESS) {
        std::cerr << "Kernel not found\n";
        return 1;
    }

    std::cout << "GPU kernel loaded successfully!\n";

    const size_t N = 128;
    const size_t bytes = N * sizeof(float);

    CUdeviceptr d_x0 = 0;
    CUdeviceptr d_x1 = 0;
    CUdeviceptr d_out = 0;

    if (cuMemAlloc(&d_x0, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_x1, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, bytes) != CUDA_SUCCESS) {

        std::cerr << "GPU memory allocation failed\n";
        return 1;
    }

    std::cout << "GPU memory allocated successfully!\n";
    std::vector<float> h_x0(N);
    std::vector<float> h_x1(N);

    for (size_t i = 0; i < N; i++) {
        h_x0[i] = static_cast<float>(i);
        h_x1[i] = static_cast<float>(i) * 2.0f;
    }
    if (cuMemcpyHtoD(d_x0, h_x0.data(), bytes) != CUDA_SUCCESS ||
    cuMemcpyHtoD(d_x1, h_x1.data(), bytes) != CUDA_SUCCESS) {

        std::cerr << "CPU to GPU copy failed\n";
        return 1;
    }

    std::cout << "Input data copied to GPU successfully!\n";

    void* kernelArgs[] = {
    &d_x0,
    &d_x1,
    &d_out
    };


    CUresult launch_status = cuLaunchKernel(
        kernel,
        1, 1, 1,       // Grid dimensions
        256, 1, 1,     // Block dimensions
        0,             // Shared memory bytes
        nullptr,       // Default CUDA stream
        kernelArgs,    // Kernel arguments
        nullptr
    );

    if (launch_status != CUDA_SUCCESS) {
        std::cerr << "Kernel launch failed\n";
        return 1;
    }

    CUresult sync_status = cuCtxSynchronize();

    if (sync_status != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        const char* error_message = nullptr;

        cuGetErrorName(sync_status, &error_name);
        cuGetErrorString(sync_status, &error_message);

        std::cerr << "GPU execution failed!\n";
        std::cerr << "Error code: " << sync_status << "\n";
        std::cerr << "Error name: "
                << (error_name ? error_name : "Unknown") << "\n";
        std::cerr << "Description: "
                << (error_message ? error_message : "Unknown") << "\n";

        return 1;
    }

    std::cout << "GPU kernel executed successfully!\n";

    std::vector<float> h_out(N);

    if(cuMemcpyDtoH(h_out.data(), d_out,bytes) != CUDA_SUCCESS){
        std::cerr<<" GPU to CPU copy failed\n";
        return 1;
    }
    std::cout << "Output copied to CPU successfully!\n";

    size_t mismatch = 0;

    for(size_t i= 0 ; i < N ; i++){
        float expected = h_x0[i] + h_x1[i];

        if (h_out[i] != expected){
            if(mismatch < 5){
                std::cerr<<"Mismatch at index " << i
                        <<": expected"<< expected<<", got "<< h_out[i]<<"\n";
            }
            mismatch++;
        }
    }

    if (mismatch == 0){
        std::cout<<"PASS: All 128 elements match!\n";
    }
    else{
        std::cerr<<"FAIL:" <<mismatch<<"incorrect elements\n";
        return 1;
    }

    cuMemFree(d_x0);
    cuMemFree(d_x1);
    cuMemFree(d_out);

    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
