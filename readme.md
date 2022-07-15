
## Overview


This repository works in tandem with the [Unity Flood Simulator](https://github.com/memeQueenPaulaDeen/UnityFloodSimulator) to fully implement a vision-based control system of a ground agent from a UAV.

This project aims to help find routes of ingress and egress after natural disasters. To this end, a UAV flies over the impacted area, collecting a global map of the disaster zone. The UAV returns to the location of the ground vehicle and helps it navigate within the environment. The process is explained in detail in the demo video below. For more information, also see [my thesis](https://vtechworks.lib.vt.edu/handle/10919/5534/browse?type=author&value=Wood%2C+Sami+Warren)

![image](https://user-images.githubusercontent.com/24756984/179283155-6de1eb02-fe18-46b7-ab03-f60a509887f3.png)(https://www.youtube.com/watch?v=vf1dAtUh1BA)



## Installation

###Download the code

Start by cloning the repository. The dependencies are managed by anaconda. If you don't already have anaconda, you will need to [download it](https://www.anaconda.com/). Once downloaded, open the "Anaconda Prompt." and CD into the directory you cloned the repo into. 

![image](https://user-images.githubusercontent.com/24756984/179303078-04087ee1-fec4-456b-8c87-1dae1dd9edc9.png)


### Create The Conda Environment

To create the conda environment by running the following command:
conda env create -f environment1.yml

![image](https://user-images.githubusercontent.com/24756984/179303316-874b6fe7-d225-455a-a947-aa03c610d14a.png)


To activate the environment type

conda activate deepLearnHWGeneral

### Download the pretrained models

The pretrained models can be downloaded from here

### Download the Training dataset

download the dataset from [only need train]:
https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH

Place the "Labeled" folder in the working directory


## Usage
