# Model Deployment with BentoML and Docker

This repository contains code and configuration files to deploy a machine learning model using BentoML and Docker. The model can be served via an API and easily containerized for scalable deployment.

## BentoML

![Screenshot 1](https://drive.google.com/uc?id=1WisOyWVejZIrQuebFNyFfzVbng0SGGJC)

![Screenshot 2](https://drive.google.com/uc?id=1UMv-B2_TeuDUNc5_wGgGlbvOKLFzoSTj)

![Screenshot 3](https://drive.google.com/uc?id=1UMv-B2_TeuDUNc5_wGgGlbvOKLFzoSTj)

![Screenshot 4](https://drive.google.com/uc?id=1eluNfA0NcfCptJJciNiCaZqkVtajm4nt)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Prepare Model with BentoML](#prepare-model-with-bentoml)
- [Build Docker Image](#build-docker-image)
- [Run Docker Container](#run-docker-container)
- [Testing the API](#testing-the-api)
- [License](#license)

## Requirements

- Python 3.7+
- Docker
- BentoML (`pip install bentoml`)
- Your trained model file (e.g., `model.pkl`, `model.pt`, etc.)
