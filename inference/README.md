# Swin-ConvLSTM for LULC Prediction

## 📌 Overview
This repository contains the implementation of a spatiotemporal deep learning framework for predicting land-use and land-cover (LULC) transitions using a hybrid **Swin Transformer–ConvLSTM** architecture.

The framework is designed for **national-scale land-use forecasting** and is evaluated using **forward temporal validation**.

## 🌍 Study Area
Malawi (Sub-Saharan Africa)

## 🧠 Model Architecture
- **Swin Transformer** → spatial feature extraction  
- **ConvLSTM** → temporal dependency modeling  
- Hybrid framework for spatiotemporal prediction  

## 📊 Models Compared
- Random Forest (RF)  
- U-Net + ConvLSTM  
- ResNet + ConvLSTM  
- Swin-CNN  
- **Swin-ConvLSTM (proposed)**  

## ⏱️ Temporal Setup
- Training: 2010–2015, 2015–2020  
- Validation: 2020 → 2024  
- Projection: 2029, 2034  

## 📁 Repository Structure
