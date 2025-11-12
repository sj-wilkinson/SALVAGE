# The SALVAGE Project Repository #

<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="left">
  <a href="https://github.com/sj-wilkinson/SALVAGE">
    <img src="SALVAGE.png" alt="Logo" width="800" height="400">
  </a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#description">Description</a>
    <li><a href="#data-access">Data Access</a>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## Introduction ##

The SDSS-ALMA Legacy-Value Archival Gas Exploration (SALVAGE) dataset is a heterogeneous but complete sampling of galaxies selected from SDSS DR7 with resolved 12CO(1-0) data in the ALMA Science Archive. The total sample is 277 galaxies across a redshift of 0.02-0.25. To image the ALMA visibilities, we use the PHANGS-ALMA reduction pipeline. Our implementation makes no modifications to the imaging pipeline itself, only fine tunes the pipeline to meet our needs using the "key files" that dictate the inputs to the pipeline.  

## Respository Structure ##

This repository is organized as follows:

| Directory | Description |
|------------|-------------|
| `/ALMA_reduction/` | Scripts used to download, calibrate, and image the ALMA data on the CANFAR Science Platform. |
| `/environments/` | Conda environment specifications to reproduce the software setup used for SALVAGE data reduction. |
| `/tutorial/` | Example Jupyter notebook demonstrating how to access, visualize, start working with SALVAGE data products. 

## Data Access ##

The reduced cubes, moment maps, and derived products can be found on the CANFAR data vault [here](https://www.canfar.net/storage/vault/list/AstroDataCitationDOI/CISTI.CANFAR/25.0077/data).

For more details, see the Accessing Data section of the [tutorial notebook](tutorial/) and the README file on the public repository.

## Acknowledgements ##

If you use SALVAGE data products or code, please cite [Wilkinson et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv251106775W/abstract).

You may also wish to cite [Leroy et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..255...19L/abstract) for use of the PHANGS-ALMA pipeline in the image processing.

## Contact ##

For questions or feedback, please contact:
Scott Wilkinson (swilkinson-at-uvic.ca)
