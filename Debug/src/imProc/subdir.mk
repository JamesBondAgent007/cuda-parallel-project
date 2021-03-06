################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/imProc/Device.cu \
../src/imProc/Device2.cu \
../src/imProc/Device3.cu 

CPP_SRCS += \
../src/imProc/Host.cpp \
../src/imProc/Image.cpp \
../src/imProc/Matrix.cpp \
../src/imProc/Point.cpp \
../src/imProc/StructuringElement.cpp 

OBJS += \
./src/imProc/Device.o \
./src/imProc/Device2.o \
./src/imProc/Device3.o \
./src/imProc/Host.o \
./src/imProc/Image.o \
./src/imProc/Matrix.o \
./src/imProc/Point.o \
./src/imProc/StructuringElement.o 

CU_DEPS += \
./src/imProc/Device.d \
./src/imProc/Device2.d \
./src/imProc/Device3.d 

CPP_DEPS += \
./src/imProc/Host.d \
./src/imProc/Image.d \
./src/imProc/Matrix.d \
./src/imProc/Point.d \
./src/imProc/StructuringElement.d 


# Each subdirectory must supply rules for building sources it contributes
src/imProc/%.o: ../src/imProc/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 -gencode arch=compute_30,code=sm_30  -odir "src/imProc" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/imProc/%.o: ../src/imProc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 -gencode arch=compute_30,code=sm_30  -odir "src/imProc" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.12_2/include/opencv -G -g -O0 -ccbin /usr/local/bin/gcc-5 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


