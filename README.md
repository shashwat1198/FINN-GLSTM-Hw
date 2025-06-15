# FINN-GLSTM-hw

This repository contains the hardware implementation of a **streamlined LSTM layer** for use with [FINN](https://github.com/Xilinx/finn), leveraging `finn-hlslib` building blocks.

## Overview

- The top-level computation is defined in **`qlstm_top.cpp`**, which represents the operations derived from the streamlined LSTM QONNX graph.
- Layer-specific parameters (size of inputs, outputs, lookbacks and datatypes) are provided in **`pipelined_lstm_header.h`**.

## Usage

The generated hardware can be exported as a **synthesizable IP block**, which integrates seamlessly with other FINN-compatible layers such as convolutional and fully connected layers. This enables the **extension of FINN to support Recurrent Neural Networks (RNNs)**, particularly LSTM-based architectures.

## Key Features

- Streamlined and synthesizable LSTM design  
- AXI-Stream interface for easy integration  
- Compatible with FINN-generated QONNX models  
- Modular IP for combining with other layers  

---
