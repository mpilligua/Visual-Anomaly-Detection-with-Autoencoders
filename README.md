# Visual Anomaly Detection with Autoencoders

[![Presentation](https://img.shields.io/badge/Presentation-Slides-blue)](https://www.canva.com/design/DAFzT9Mw-dk/1hTZf2D5NbSwaLdFb-X5_g/edit?utm_content=DAFzT9Mw-dk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

This repository hosts an implementation of a visual anomaly detection system using autoencoders, specifically tailored for detecting the presence of Helicobacter pylori (H. pylori) in histological images with immunohistochemical staining. H. pylori is a bacterium classified as a class 1 carcinogen to humans since 1994, making its detection crucial for early diagnosis and treatment.

## Motivation 
Traditional methods for detecting H. pylori involve labor-intensive manual inspection of digitized histological images by expert pathologists. This process is time-consuming and subjective, leading to potential errors. Our motivation is to develop an automated system that can accurately and efficiently detect H. pylori in histological images, thereby aiding pathologists in their diagnostic workflow.

## Methodology
We propose the use of autoencoders, a type of neural network architecture, to learn latent patterns of healthy tissue and identify anomalies indicative of H. pylori infection. Unlike traditional classification approaches, autoencoders can learn patterns in a self-supervised manner, eliminating the need for extensive image annotations. 
Once the intrinsic representation of healthy tissue is learned we can use this model to try to reconstruct unlabelled images and based on the reconstruction error classify this patch as infected or healthy. 

## Data 
We trained using a dataset of 

