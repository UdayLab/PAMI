[__<--__ Return to home page](index.html)

## Maintenance of PAMI library

We first describe the basic requirements for installing and using PAMI library. Next, we present the maintenance of PAMI library using pip command.
Next, we describe the cloning of PAMI library from GitHub using _git_ command.  Finally, we discuss the methodology to 
download the Zip file of PAMI from GitHub and use it.

### Requirements
1. Python version: 3.5 and above
2. Operating system: any
3. Pre-requisites: psutil and pandas

### Approach 1: Using pip command
This is the simplest and convenient way to install, upgrade, and uninstall the PAMI library.  

1. Installation step 

       pip install pami

2. Upgrade step

       pip install pami -U

3. Uninstallation step  

        pip uninstall pami -y

4. Seeing the installation details
    
       pip show pami

[**CLICK HERE**](https://pypi.org/project/pami/) for more information on installing PAMI using pip.

### Approach 2: Cloning through GitHub
Installation of PAMI library in some universities and industries may require prior permission from the administrative 
departments. In such cases, _cloning_ may found to be a convenient way to maintain the latest version of PAMI library. 

1. Clone the PAMI repository using git command
   
       git clone https://github.com/udayRage/PAMI.git

2. Regularly update the PAMI repository

       git pull PAMI

3. Delete the PAMI repository

       git branch -d PAMI

### Approach 3: Downloading the library from the GitHub
This is the most difficult and cumbersome process to utilize and maintain PAMI library. Thus, we recommend one of the above two process for utilizing the PAMI library.
However, we are providing this process of installation for completeness purposes.

As PAMI is a python package, users can download the source code of this library from the Github and use it. We now explain this process.

1. Download the PAMI-main.zip file from [GitHub](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip)
2. Unzip the PAMI-main.zip file.
3. Enter into the PAMI-main folder and move the PAMI folder to the location of your choice. The location of your choice can be any folder in your machine or the folder of your source code.