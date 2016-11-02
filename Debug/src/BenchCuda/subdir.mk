################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/BenchCuda/BenchCuda.cu 

OBJS += \
./src/BenchCuda/BenchCuda.o 

CU_DEPS += \
./src/BenchCuda/BenchCuda.d 


# Each subdirectory must supply rules for building sources it contributes
src/BenchCuda/%.o: ../src/BenchCuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 -gencode arch=compute_30,code=sm_30  -odir "src/BenchCuda" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


