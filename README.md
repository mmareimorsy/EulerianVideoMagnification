# EulerianVideoMagnification

The folder structure should contain the following files & folders:

    * build.py
    * eulerian_magnification.py
    * phase_based_magnification.py
    * common.py
    * parameters_high_alpha.json
    * parameters_low_alpha.json
    * parameters_mid_alpha.json
    * samples folder
    * requirements.txt

# File contents

* build.py
This is the main code to run for doing both Eulerian & Riesz pyramids phase based magnification

* eulerian_magnification.py:
This file contains the code needed to do Eulerian Video Magnification process

* phase_based_magnification.py
This file contains the code needed to do the phase based magnification approach using Riesz pyramids

* common.py
This file has some helper functions that is commonly used, things like extracting video frames, saving frames, building pyramids, etc..

* .json files
The three json files contain the video file names, with the magnification & frequency range parameters used for building the results provided

* samples folder:
This is where the code looks for the video files, video file names should match what is given in the json files

* requirements.txt:
This is a snapshot of the libraries & versions used, this could be used as 
python3 -m pip install -r requirements.txt

# JSON files

The json files are structured as:

    magnificationMethod{

        videoFileName{

            magnification parameters filled in here
        
        }
        
    }

# How to get the results & what to expect

To run the code:

python3 build.py

following folder structure would be created in the working directory
* results: This is a parent folder for all the results
    * eulerian: This folder contains the results from running the intensity based approach 
    * phaseBased: This folder contains the results from running the phase based approach
        * under each of the above folders:
            * magnified videos will be stored here, for example "baby_Eulerian_magnified_alpha8.mp4", that's the original file name followed by the method used & the alpha value used
            * subfolder would be created for each video file processed
                * subfolder named with the original video file name ; example "baby"
                    * subfolder named "videoFramesOrigin" : Those are the video frames from the original video
                    * subfolder named "magnifiedFrames" : Those are the frames used to construct the magnified video

Notes about testing:
* Results have been generated on a core-i5 intel 6th generation CPU with 16GB RAM
* On average each video file consumes around 3 to 4 GB RAM; some code refactoring was done to reduce RAM consumption but there is still room for improvement
* To build the results as provided in link provided in the project report it typically takes around 2 hours or possibly less on faster machines 