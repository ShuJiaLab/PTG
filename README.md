# PTG
Hyperspectral Phasor Thermography 
============

Description
-----------
The software package provides the codes for Phasor Thermography, which includes the hyperspectral image registration, thermal phasor transformation, Phasor-Enabled Multiparametric Thermal Unmixing(PEMTU), and phasor analysis and visualization.

Table of Contents
-----------------
- Installation
- Usage
- Features
- License
- Contact

Installation
------------
Instructions on how to install the project. Include steps for setting up any dependencies.

1. Install MATLAB for image registration, Phasor transformation, PEMTU algorithm

  https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html

2. Install Python and create the required environment using the phasorAnalysis_Visualization.yaml file with the codes below:
          conda env create -f my_env.yaml
 


Usage
-----
A guide on how to use the codes. Include examples of commands.

1. Image registration codes run in folder named 'ImageRegistration'.

2. Run 'Thermal_Phasor_transformation.m' to conduct phasor transformation for the hyperspectral thermal image stack

3. The phasor analysis and visualization is achieved by running 'Phasor_Analysis_Visualization.ipynb' in folder named 'Phasor_Analysis_and_Visualization'.


4. Run the codes in MATLAB for PEMTU:
   run Face_Phasor_Main.m in  folder named'PEMTU'
 

Features
--------
Please follow the four steps to achieve the hyperspectral phasor thermography: 

1. Hyperspectral image registration

2. Phasor transformation

3. Phasor Analysis and Visualization

4. Phasor-Enabled Multiparametric Thermal Unmixing algorithm


Contact
-------

- Dingding Han - dhan314@gatech.edu   Shu Jia -shu.jia@gatech.edu
- GitHub - https://github.com/ShuJiaLab/PTG
