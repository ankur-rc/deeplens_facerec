# Face Trigger ![lambda](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5CLARGE%20%5Clambda)
``` Inference lambda function for DeepLens```

This repository houses the requisite code to run real-time face recognition using the [Face Trigger](https://github.com/SofturaInternal/face-trigger) library.

![screenshot](https://github.com/SofturaInternal/face-trigger-lambda/blob/master/imgs/screenshot.png)

## Setting up DeepLens
1. Follow the [getting started](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-getting-started.html) guide.

2. Create a virtual environment ([virtualenv](https://virtualenv.pypa.io/en/stable/)) for the python (2.7) project.
  - Activate the virtual environment.

3. Create a folder and clone the repository into the folder.

4. Open a terminal window and navigate to the project root.

4. Install CMake (required for compiling _dlib_)
  - ```bash 
      sudo apt-get install cmake
    ```

5. Install the python dependencies.
  - ```bash
    pip install -r requirements.txt
    ```
6. Run the program:
  - ```bash 
    python webcam_facerec.py 
    ```
## Run as service (Optional)
The app can be run as a service on boot.

1. Copy the file _face\_trigger.service_ to _/etc/systemd/system

2. Enable the service to run on boot
  - ```bash
    systemctl enable face_trigger
    ````
3. Install espeak (for offline tts)
  - ```bash
    sudo apt-get install espeak
    ```
4. Check for the sound card sinks by running the following command:
  - ```bash 
    pacmd list-card
    ```
  - Look out for the section in the output that lists the sink names: 'alsa_output...monitor'
  - Copy the sink that ends in:
    - 'headphone' for sound output from 3.5mm 
    - 'HDMI' for output from hdmi port
    
5. Edit the file _/etc/pulse/default.pa
  - Change the line that starts with 'set-default-sink XXX'
  - Change XXX to the value copied in step 4.`
  - Save the file.
  
6. Reboot. Give the system some time to bootup and finalise initialising the services (~5mins)
