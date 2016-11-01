################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/convolutionBench/BenchManager.cpp \
../src/convolutionBench/IConvBench.cpp 

OBJS += \
./src/convolutionBench/BenchManager.o \
./src/convolutionBench/IConvBench.o 

CPP_DEPS += \
./src/convolutionBench/BenchManager.d \
./src/convolutionBench/IConvBench.d 


# Each subdirectory must supply rules for building sources it contributes
src/convolutionBench/%.o: ../src/convolutionBench/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src/convolutionBench" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


