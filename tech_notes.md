# Technical Notes for sam3-2026-01-31

## Jupyter notebooks

Metadata extracted using `ls -lt *.ipynb`

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
### Feb 24 16:50 rtest.ipynb
### Feb 24 15:38 visualizer.ipynb
### Feb 18 13:24 pca.ipynb
### Feb 18 09:10 processing_videos.ipynb
### Feb 14 07:33 sqlite_tables.ipynb
### Feb  7 08:50 wkt.ipynb
### Feb  7 05:00 efd2.ipynb
### Feb  6 06:56 efd.ipynb

## Python scripts

### Mar  6 08:26 pyefd_example.py
### Mar  3 17:28 am_ptools_tests.py
### Mar  3 17:25 am_ptools.py
### Feb 27 14:30 roadside.py
### Feb  8 08:20 test_roadside.py
### Feb  8 06:58 efd2.py
### Jan 31 14:07 main.py
