RGBanalysis
===========

## Description

Codes for analysing picture data from film degradation chamber (https://github.com/PV-Lab/hte_degradation_chamber). Analaysis with and without color calibration.

The code assumes that there is a sample holder with 4 (horizontal rows in the picture area) x 7 (vertical columns in the picture area) samples and a small reference color chart with 6x4 color patches in the picture area during the aging test. It also assumes the first picture of the aging test is of Xrite ColorChecker Passport (a reference color chart with 6x4 color patches)

## Use instructions

The code produces both color calibrated and raw color data as a function of measurement time. The use of color calibration is essential for ensuring repeatable and reproducible results. Approach implemented is described in detail in the ESI of the following article: https://doi.org/10.1016/j.matt.2021.01.008 

Open Main.py and modify according to the instructions in the file. If necessary, use Test_crop_box.py for adjusting crop boxes (for slicing image files into 28 individual samples, a small reference color chart, and Xrite Colorchecker Passport chart).

An example of the output files is shown in folder 'Results of optical analysis'.

Video feature has been tested only in Linux and should be commented out in Main.py if using in Windows.

The picture of Xrite ColorChecker Passport is used for color calibration of the rest of the pictures during the aging test. One can rely on this calibration as long as the photographed color of the small reference color chart (that is in the picture area during the whole aging test) remains constant.


## Installation

To install, just clone the following repository and install the packages prompted when running the codes:

`$ git clone https://github.com/PV-Lab/RGBanalysis.git`

`$ cd RGBanalysis`

To activate the video creating feature, use in Linux, install ffmpeg (https://github.com/FFmpeg/FFmpeg) and run Main.py normally.

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen, Zhe Liu, Siyu I.P. Tian | 
| **VERSION**      | 1.0 / Sep, 2020 | 
| **EMAILS**      | armi.tiihonen@gmail.com, chris.liuzhe@gmail.com, isaactsy777@hotmail.com  | 
||                    |


## Attribution

Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation

    @Misc{rgbanalysis2020,
      author =   {The RGBanalysis authors},
      title =    {{RGBanalysis}: Camera degradation data analysis with color calibration},
      howpublished = {\url{https://github.com/PV-Lab/RGBanalysis}},
      year = {2020}
    }
    
    Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter, 2021, https://doi.org/10.1016/j.matt.2021.01.008.
