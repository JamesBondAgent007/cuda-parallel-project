################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Device.cu \
../Device2.cu \
../Device3.cu 

CPP_SRCS += \
../Host.cpp 

OBJS += \
./Device.o \
./Device2.o \
./Device3.o \
./Host.o 

CU_DEPS += \
./Device.d \
./Device2.d \
./Device3.d 

CPP_DEPS += \
./Host.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


