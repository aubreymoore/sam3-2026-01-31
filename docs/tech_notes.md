---
title: Technical Notes for ~/Desktop/sam3-2026-01-31
author: "Aubrey Moore"
date: "2026-03-14"
# geometry: "a4paper, margin=1.5cm"
# documentclass: article
# fontsize: 12pt
# urlcolor: blue
# toc: true
exports: ["pdf"]
---

# Technical Notes for sam3-2026-01-31

```{warning}
We believe in a community-driven approach of open-source tools that are composable and extensible.
```
````{note}
To convert of this file to a PDF with an index:
```
pandoc tech_notes.md -o tech_notes.pdf
```
````

## Jupyter notebooks

````{note}
I extracted metadata extracted for this section using
```
ls -lt *.ipynb
```
````

### Mar 10 05:06 poly2mask.ipynb

- Calculates EDFs from detection polygons stored in SQLite db
- NEEDS CLEANING
- functions 
    - optional_plotting_example
    - polygon_to_mask2
    - get_mask_contour
- main
    - gets data from sam3_detections_sqlite3
    - for each detection: 
        - converts polygon to mask
        - calculated contour around mask
        - calculates elliptic Fourier descriptor for contour

### Feb 27 19:36 extract_pdf_highlights.ipynb

- PROBABLY BETTER TO SAVE RESULTS IN CSV
- extracts annotations from a marked up PDF and saves results in a database

### Feb 27 15:05 2026-02-14.ipynb

- MAIN WORKFLOW
- for each image:
    - gets detections using `rs.run_sam3_sematic_predictor()`
    - saves image data (path, width, height, time, latitude, longitude) to SQLite table named `images` using `rs.get_data_for_images_table()`
    - for each detection:
        - saves detection data (class, polygon, bounding box, confidence) to SQLite table named `detections` using
        `rs.get_data_for_detections_table`

### Feb 27 09:51 sam3-2026-01-31.ipynb

- replaced by 2026-02-14.ipynb?

### Feb 24 16:50 rtest.ipynb

- evaluation of running the R Momocs library using `rpy2`
- I decided to use `pyefd` instead of the `Momocs` R library

### Feb 24 15:38 visualizer.ipynb

- This Jupyter notebook visualizes detections of coconut palms in images using data from an SQLite database.

- Each detection is visualized using a bound box and polygon overlay applied to the original image.

- Report generation
    -This notebook generates a markdown report which can be converted to a PDF using `pandoc`.
    This PDF can then be used to create metadata for images using `okular` or some other PDF editor.
    The idea is to hightlight keywords associated with images.

### Feb 18 13:24 pca.ipynb

- This notebook explores principal component analysis of elliptic fourier descriptors to differentiate shapes of coconut palms detected in images.

### Feb 18 09:10 processing_videos.ipynb

- This notebook evaluation of `SAM3VideoSemanticPredictor` to scan videos from roadside surveys of CRB damage. Using videos instead of still images may me advantageous in handling occluded objects.

### Feb 14 07:33 sqlite_tables.ipynb

- superceded by `2026-02-14.ipynb`?

### Feb  7 08:50 wkt.ipynb

- legacy?

### Feb  7 05:00 efd2.ipynb

- superceded by `poly2mask.ipynb`?

### Feb  6 06:56 efd.ipynb

- superceded by `efd2.ipynb`?

## Python scripts

### Mar  6 08:26 pyefd_example.py

- An example showing how to use pyefd for fitting points along a closed curve.

### Mar  3 17:28 am_ptools_tests.py

- Creates a matplotlib figure with a list of plots arranged in a specified number of columns.

### Mar  3 17:25 am_ptools.py

- Unit tests for `am_ptools.py`

### Feb 27 14:30 roadside.py

- IMPORTANT
- Python modules containing functions for my CRB roadside surveys project

### Feb  8 08:20 test_roadside.py

- test code for `roadside.py`

### Feb  8 06:58 efd2.py

- "raw" python code for EFD
- I don't remember where I got this
- may be better to stick with source code for `pyefd`

## Lit search

#### "elliptic fourier descriptors" "deviation analysis"

https://share.google/aimode/SoLZS1m3UGNv6EbGS

Finding direct, permanent PDF links for academic papers can be tricky due to paywalls and shifting URL structures. However, I have compiled the list below using **DOI (Digital Object Identifier)** links or **PubMed/Open Access** links, which are the most reliable ways to access the full-text versions.

For several of these, you may need institutional access, but many (marked as **Open Access**) are freely available to the public.

---

### Citations with Full-Text Links

1. **Bonhomme, V., et al. (2014).** "Momocs: Outline Analysis Using R." *Journal of Statistical Software*.
**[[Open Access PDF]](https://www.jstatsoft.org/article/view/v056i13/v56i13.pdf)**
2. **Ge, X., et al. (2021).** "A Deep Learning-Based Method for Fruit Shape Phenotyping in Strawberry." *Frontiers in Plant Science*.
**[[Open Access PDF]](https://www.google.com/search?q=https://www.frontiersin.org/articles/10.3389/fpls.2021.631835/pdf)**
3. **Kuhl, F. P., & Giardina, C. R. (1982).** "Elliptic Fourier features of a closed contour." *Computer Graphics and Image Processing*.
**[[Full Text via ScienceDirect]](https://doi.org/10.1016/0146-664X(82)90034-X)** *(Requires login/purchase)*
4. **Li, M., et al. (2020).** "Analysis of Leaf Morphological Characters of *Zelkova serrata* Based on Elliptic Fourier Descriptors." *Forests*.
**[[Open Access PDF]](https://www.google.com/search?q=https://www.mdpi.com/1999-4907/11/8/875/pdf)**
5. **Martinez-Abadias, N., et al. (2018).** "Quantifying facial shape variation in Chagas disease using Elliptic Fourier Descriptors." *PLOS Neglected Tropical Diseases*.
**[[Open Access PDF]](https://www.google.com/search?q=https://journals.plos.org/plosntds/article/file%3Fid%3D10.1371/journal.pntd.0006871%26type%3Dprintable)**
6. **Migicovsky, Z., et al. (2018).** "Patterns of Genotype-Phenotype Associations in Apple." *New Phytologist*.
**[[Full Text via Wiley Online Library]](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.14627)**
7. **Sievänen, J., et al. (2022).** "Automatic identification of fish species using shape analysis of scales." *Fisheries Research*.
**[[Full Text via ResearchGate]](https://www.google.com/search?q=https://www.researchgate.net/publication/358170068_Automatic_identification_of_fish_species_using_shape_analysis_of_scales)**
8. **Smith, A., & Williams, J. (2019).** "Quantifying the Morphology of Prehistoric Stone Tools through Elliptic Fourier Analysis." *Journal of Archaeological Science: Reports*.
**[[Full Text via DOI]](https://www.google.com/search?q=https://doi.org/10.1016/j.jasrep.2018.11.026)**
9. **Wang, H., et al. (2023).** "Grain shape characterization of calcareous sand based on Elliptic Fourier Descriptors." *Bulletin of Engineering Geology and the Environment*.
**[[Full Text via Springer]](https://www.google.com/search?q=https://link.springer.com/article/10.1007/s10064-023-03112-w)**
10. **Zhang, C., et al. (2022).** "DeepMorph: A high-throughput software for zebra fish scale morphometrics." *Methods in Ecology and Evolution*.
**[[Open Access PDF]](https://besjournals.onlinelibrary.wiley.com/doi/pdf/10.1111/2041-210X.13943)**

---

### Quick Tip for Access

If you encounter a paywall for any of the non-Open Access links, you can often find a legal "Author's Original" or "Pre-print" version by searching the title on [Google Scholar](https://scholar.google.com) and looking for the **[PDF]** link on the right-hand side of the search results.

**Would you like me to help you format these into a specific bibliography style, such as APA or BibTeX?**



## Sphinx documentation

#### Installed sphinx as a global tools using
```
uv run install sphinx
```
#### Check installation
```
sphinx-build --version
```
#### Add a sphinx docs directory by executing this command from the top directory of the project
```
sphinx-quickstart docs
```
#### Create html files
```
cd docs
make html
```
Sphinx docs for the project can now be viewed by opening `docs/_build/html/index.html` in a web browser.

#### Edit `docs/conf.py` to look something like this example.
```python
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

# Add the project root directory to sys.path
sys.path.insert(0, "/home/aubrey/Desktop/lumache/")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Lumache"
copyright = "2026, Aubrey Moore"
author = "Aubrey Moore"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

```

#### Create a `docs/usage.rst` file using this example
```
Usage
=====

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients()
.. autofunction:: lumache.add_numbers()

```

#### Add a link to `docs/usage.rst` in `docs/index.rst`
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
```

#### Regenerate html
```
cd docs
make html
```
If `make html` fails with errors, fix them and try again.

For example, when `exception: No module named 'sphinx_autodoc_typehints'` was reported,
I simply executed fixed this using `uv add sphinx-autodoc-typehints`.

`roadside.py` IS IN 2 PLACES: PROJECT ROOT AND `src/roadside`. MUST BE FIXED.

## Ruff

https://youtu.be/828S-DMQog8
