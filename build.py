import json 
from eulerian_magnification import *
from phase_based_magnification import *

fileTrials = ["parameters_mid_alpha.json", "parameters_high_alpha.json", "parameters_low_alpha.json"]

# fileTrials = ["testSample_1-3.json","testSample_2-4.json","testSample_4-6.json","testSample_6-8.json"]

for jsonFile in fileTrials:
    try:
        parametersFn = open(jsonFile, "r")
    except FileExistsError:
        print("Please make sure parameters.json file exist in current directory")

    parameters = json.load(parametersFn)

    for magnifyMode, files in parameters.items():
        if magnifyMode == "eulerian":
            print("working on Eulerian based magnification")
            for fileName, params in files.items():
                if params["filter"] == "butter":
                    try:
                        videoMagnificationButterWorthFilter(fileName,params["alpha"],params["lambda_c"],params["samplingRate"],params["chromeAttenuation"],params["lowFreq"],params["highFreq"])
                    except Exception as e:
                        print("Failed in processing", fileName, "due to", e)

                elif params["filter"] == "ideal":
                    try:
                        videoMagnificationIdealFilter(fileName,params["alpha"],params["lambda_c"],params["samplingRate"],params["chromeAttenuation"],params["lowFreq"],params["highFreq"])
                    except Exception as e:
                        print("Failed in processing", fileName, "due to", e)
        elif magnifyMode == "riesz":
            print("working on Riesz based magnification")
            for fileName, params in files.items():
                try:
                    phaseBasedMagnification(fileName, params["alpha"],params["samplingRate"], params["lowFreq"], params["highFreq"])
                except Exception as e:
                    print("Failed in processing", fileName, "due to", e)