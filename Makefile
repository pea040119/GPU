# File: 		Makefile
# Created: 		2025-08-27
# Updated: 		2025-08-29 by PEA
# Description: 	Makefile for building and managing CUDA projects.


# Usage:
#   make					(default) build + ptx + sass
#   make build				Build executable
#   make run				Run executable
#   make obj				Compile object only
#   make ptx				Dump PTX from object
#   make cubin				Generate .cubin
#   make sass				Disassemble cubin to SASS
#   make gdb				Debug with cuda-gdb
#   make clean				Remove build artifacts

# Config:
#   TARGET=(file_name)		Source: src/$(TARGET).cu  Output dir: build/$(TARGET)/
#   SM=(GPU architecture)	GPU SM architecture (e.g. 70,75,80,86,89)
#   RDC=(0/1)				Enable relocatable device code (default 0)



# Config
TARGET				:= heap_overflow
GDB					?= 0

SRC_DIR				?= src
BUILD_DIR			?= build
BUILD_TARGET_DIR	:= ${BUILD_DIR}/${TARGET}
SRC_FILE			?= ${TARGET}.cu
OUTPUT_FILE			?= ${TARGET}.out
OBJ_FILE			:= ${TARGET}.o
PTX_FILE			:= ${TARGET}.ptx
SASS_FILE			:= ${TARGET}.sass
CUBIN_FILE			:= ${TARGET}.cubin

# Toolchain
NVCC        		?= nvcc
CUOBJDUMP   		?= cuobjdump
NVDISASM    		?= nvdisasm
CUDA_PATH   		?= /usr/local/cuda

# GPU Architecture
SM       			?= 75 # RTX 2060 Super
GENCODE				:= -arch=sm_$(SM)


# Debugging Options
RDC ?= 0
ifeq ($(RDC),1)
  RDC_FLAG	 		:= -rdc=true
else
  RDC_FLAG 			:=
endif

WARN_HOST 			:= -Wall -Wextra
FRAMEPTR  			:= -Xcompiler -fno-omit-frame-pointer
PTXAS_V   			:= -Xptxas -v
LINE_INFO 			:= -lineinfo

DBG_HOST  			:= -g -O0
DBG_DEV   			:= -G -O0 -Xptxas -O0
DBG_MORE  			:= -Xcompiler "-g3 $(WARN_HOST)" $(FRAMEPTR) $(PTXAS_V)

NVFLAGS_BASE 		:= $(GENCODE) $(RDC_FLAG) $(LINE_INFO)
ifeq ($(GDB), 1)
	NVFLAGS   			:= $(NVFLAGS_BASE) $(DBG_HOST) $(DBG_DEV) $(DBG_MORE)
else
	NVFLAGS   			:= $(NVFLAGS_BASE)
endif


.PHONY: all prepare build obj ptx sass cubin run clean

all: build ptx sass


prepare:
	if [ -d $(BUILD_TARGET_DIR) ]; then rm -rf "$(BUILD_TARGET_DIR)"; fi
	mkdir -p $(BUILD_TARGET_DIR)


obj: prepare
	$(NVCC) $(NVFLAGS) -c -o $(BUILD_TARGET_DIR)/$(OBJ_FILE) $(SRC_DIR)/$(SRC_FILE)


build: obj
	$(NVCC) $(NVFLAGS) $(BUILD_TARGET_DIR)/$(OBJ_FILE) -o $(BUILD_TARGET_DIR)/$(OUTPUT_FILE)


ptx: obj
	$(CUOBJDUMP) --dump-ptx $(BUILD_TARGET_DIR)/$(OBJ_FILE) > $(BUILD_TARGET_DIR)/$(PTX_FILE)


cubin: prepare
	$(NVCC) $(NVFLAGS) -cubin $(SRC_DIR)/$(SRC_FILE) -o $(BUILD_TARGET_DIR)/$(CUBIN_FILE)


sass: cubin
	$(CUOBJDUMP) --dump-sass $(BUILD_TARGET_DIR)/$(CUBIN_FILE) > $(BUILD_TARGET_DIR)/$(SASS_FILE)


run: 
	$(BUILD_TARGET_DIR)/$(OUTPUT_FILE)


gdb:
	cuda-gdb $(BUILD_TARGET_DIR)/$(OUTPUT_FILE)


clean:
	rm -rf "$(BUILD_TARGET_DIR)"