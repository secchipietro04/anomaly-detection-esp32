# Inference with LSTM Autoencoder for Anomaly Detection on ESP32

## Project Overview
This project, developed during a university internship at **IOTINGA S.r.l**, explores the application of a Long Short-Term Memory (LSTM) Autoencoder for anomaly detection using gyroscope data. The primary objective is to deploy a machine learning model on an **ESP32 microcontroller** for real-time anomaly detection, demonstrating how edge computing can enable advanced analytics on low-power devices.

## Key Features
- **Data Collection**: Gyroscope data is acquired from the ESP32's onboard sensors or external IMU modules.
- **Model Training**: An LSTM Autoencoder model is trained on vibration data.
- **Anomaly Detection**: The model performs inference on the ESP32, flagging deviations as anomalies.


## Components
### Hardware
- **ESP32**: (ESP32-S3 used in testing) Microcontroller.
- **Gyroscope Sensor**: (steval-mki245ka used in testing) IMU sensor for gyroscope readings.


### Software
- **Python**: For model training and preparation.
- **TensorFlow/Keras**: Used to develop and train the LSTM Autoencoder model.
- **TensorFlow Lite Micro**: For converting the trained model into a lightweight format suitable for ESP32.
- **ESP-IDF**: For programming the ESP32 and deploying the model.

## Workflow
1. **Data Collection**

2. **Model Training**

3. **Model Convertion**

4. **Model loading**:
   - Load the TensorFlow Lite model onto the ESP32.

5. **Inference**

## Limitations
    Inference time is roughly 5000 microseconds/200 Hz so the microcontroller cannot do real time calculations if the sampling frequency it too high. The sensor and the i2c communication allow for one read every 420 microseconds/2380Hz so the ratio is roughly 12:1 
## Future Work
- Better sessions handling, both on client side and server side.
- Trained model evaluations with test datasets.
- Explore alternative model architectures.
- Find a viable model architecture that does not need for further training on deployment.


## Acknowledgments
This project was made possible through the support and resources provided by **IOTINGA S.r.l** and **Università di Verona** during the internship period.


## Contributors
   - **Secchi Pietro Giampaolo** - Intern at IOTINGA s.r.l.

## Special thanks to
   - **Simone Camporeale**  Co-founder of IOTINGA S.r.l
   - **Matteo Bissoli**     Co-founder of IOTINGA S.r.l
   - **Alessandro Righi**   my tutor at IOTINGA S.r.l
