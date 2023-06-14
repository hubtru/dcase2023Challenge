# Dcase 2023 Challenge

This project is based on the [Dcase 2023 Challenge](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring). 
The goal is to achieve a generative model for machine condition monitoring.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Project is build with Python 3.10. Install the project as followed:

```bash
$ git clone https://github.com/your-username/your-project.git
$ cd your-project
$ pip install -r requirements.txt
```

## Usage
First, edit `audio_all` to the path of your dataset root folder. The file structure must be like this: 


- `audio_all/` (root directory)
  - `bearing/` (subdirectory)
    - `train/` (nested subdirectory)
      - `section_00_source_train_normal_0000_vel_8_loc_B.wav`
      - ...
      - `section_00_target_train_normal_0009_vel_16_loc_E.wav`
    - `test/` (nested subdirectory)
      - `document1.pdf`
      - `document2.docx`
  - `fan/` (subdirectory)
    - `train/` (nested subdirectory)
      - ...
    - `test/` (nested subdirectory)
      - ...
  - ...

`feature_options` shows all available features_types. The feature is set with `feature`. 
The output_size of the feature_array can be declared in `output_size`. 

In the first run it is currently necessary to use `compute_all_features()`. 
It creates .json files for the features for each dataset and subset separately. 
The files are saved here:
- `dcase2023Challenge/` (root project directory)
  - `data_features/` (subdirectory)
    - `stft_valve_train_128_313.json` 
    - ...

The files will then be loaded with `load_all_features()`. 
The subdirectories can be specified in `datasets`, the nested subdirectories in `subsets`. 
After the first run you only need `load_all_features()` to work with the existing files.  