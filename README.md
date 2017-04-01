# finalyearproject
Hugo, Sigurd, Kevinn and Veronica's Final Year Project @ HKUST 2017
Six Sigma Green Belt project for X Company


## GUI.py:
> Update 03/20/2017: Partial implementation of reprocessing for single lines (S/A code).

> Update 03/12/2017: Implemented accuracy test (for the whole box) for boxes with only printed characters. Please use "MasterASN_3" to carry out such tests. Improved output to text files.

> Update 03/11/2017: Exception handling

> Update 03/09/2017: Added test option for the Hybrid method 

> Update 03/03/2017: Further improvements to matching logic. Can now correctly identify the item code in ≈ 98% of samples, as long as there is an ASN. 

> Update 02/21/2017: Added implementation of Wagner-Fischer algorithm, which is used for string processing. Improved the hybrid method (binarization). Improved the logic for categorizing lines.

> Update 02/17/2017: Added adaptive thresholding, you can select it from the drop-down menu in the GUI.

> Update 02/16/2017: Added reprocessing for "snowy" outputs for the Hybrid method, which dramatically improves the outputs for boxes with rough background. 

> Update 02/12/2017: The binarization technique introduced in Hybrid.py has been updated, the processing time has been dramatically decreased. Work in progress: Matching, extraction of single signiciant digits in solid code for KNN classification.

> Update 02/08/2017: Added hybrid.py to the GUI, and started work on interpreting text outputs from Tesseract.

Simple GUI, making it more convenient to test our program on sample images.
Uses Tkinter, so it should run on both Mac and Windows.

## Segmentation.py: 
> Update 04/01/2017: Added KNN classification for all the candidate cuts and the hypothesis that no cuts are needed. Results are surprisingly good. 

Takes a binary image of a single line (solid or assort), and carries out segmentation of touching characters. Is still work in progress. Based on a paper by Fujisawa et al.

## Adaptive_thresholding.py: 
A less computationally expensive approach than hybrid.py. Apart from using skimage's adaptive thresholding method, contour analysis is used to filter out any noise left after thresholding.


## CategorizeLines.py: 
Uses OCR output as input. Returns the indeces for the useful lines. Does not rely on ASN, but is specific to one particular upstream company.


## Hybrid.py: 
> Update 02/24/2017: Is now up-to-date. 

> Update 02/06/2017: Approaching perfection. Needs a small dose of dilation, a touch of line rotation and a remedy for boxes covered in excessive amounts of tape. 
Implementation of the paper mentioned (with a bunch of modifications). Is processing images with complex backgrounds. 



## knn.py: 
Based on http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html and http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html.
Used for classificaiton of handwriting.
Tried with 20,000 training data and 20,000 test data (digits 0-9) and achieved 97.81% accuracy with parameter k = 5. 
Please use the pickles from our google drive as data.
The code takes a few minutes to run if you have lots of data. 

## Extractandmergelines.py: (outdated, See GUI.py)
Pre-processes images, and feeds them to Tesseract. The OCR output is interpreted. NOTE: in order to use, please create an empty txt file, output.txt in your project folder. Also, download ASN Example.csv from Google Drive. No longer assumes the solid/assort code is below the item code, can also deal with cases where the codes are on the same line.

Command line format (using Terminal): ```tesseract pic.png output``` then to view ```open -e (or -a "app_name") output.txt```

> Update 12/30/2016: Try in Python IDE instead of in bash shell

## Charextraction.py:(outdated, update coming)
Finds the locations of the hyphens in the Solid code. Based on this, each character from the Solid code is extracted, and saved on a 28x28 black background. The images can then be fed to tensorflow (or something else) to be read. To use, please first run extractandmergelines.py in order to generate the necessary input file. Future improvements: separation of touching handwritten characters, limiting the number of characters selected to 8 (ab-cde-fgh).


## Benchmark.py:
Use to simply use tesseract OCR on any image - will usually not be very accurate

## Histograms.py:
Might be useful later. Is related to this paper (about segmenting handwriting): http://www.ee.bgu.ac.il/~dinstein/stip2002/Seminar_papers/David_Cahana_A%20character%20segmentation%20method%20using%20projection%20profile-based%20technique.pdf 

## Interpretoutput.py: (outdated)
Given a txt file containing the OCR output from the initial pre-procesing and an advance shipment notice (ASN) in csv format (see google drive), it categorizes each line of text into the following categories:
Item code, solid (assort) code, quantity, box number, item description, or "nothing". 
This will be useful for determining whether further image processing is needed for a particular input image. 

## Letteraccuracy.py: 
Using data from http://ai.stanford.edu/~btaskar/ocr/, for checking accuracy of reading single, handwritten, lowercase letters (a-z). Uses Tesseract, but you can replace it with whatever tool you want to test. To use this code, download the data from http://ai.stanford.edu/~btaskar/ocr/ and place it in your project folder. The number of data points is >50000, so the processing time is a bit long.

## Count.py:
Simple script to calculate the # of correctly identified chars and the # of incorrectly identified chars. Manipulate the string ```digit``` to change output. Default input file is ```output.txt```

## What needs to be done: 
 - Find data set of segmented (single) letters A-Z (COMPLETED) and a-z (Dataset w/ larger resolution would be nice, the one mentioned above has 16x8).

 > Update 1/6/2017: Carried out all the requested handwriting accuracy tests w/ Tesseract

 > Update 12/31/2016: I uploaded a decent amount of segmented, handwritten, samples to our google drive (>400,000 digits and 800,000 characters (A-Z, a-z and 0-9)). Please check the "Pickled_data" folder to access the data.
 
 > Update 12/29/2016: I uploaded English_Handwritten_Pngs with single letters (A-Za-z, 55 samples each) found here <http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/>
 

 - Find an appropriate way of reading handwritten characters.
 
 > Update 1/6/2017: See knn.py for high success rate (digits), will test on letters soon.
 
 > Update 12/27/2016: Tried Tesseract with grayscale PNGs of handwritten digits made available at <http://www.cs.nyu.edu/~roweis/data.html>, success rate between 5-75%

 > Update 12/30/2016: Updated %accuracy using MNIST and USPS grayscale JPGs
 
 > To try/trying: use jTessBoxEditor (Java-based box editor) <http://vietocr.sourceforge.net/training.html> to train Tesseract for different fonts/handwriting -- make sure you have Java installed & ready to compile

 - Implement a way to use histograms for segmentation and cleaning (NOT PRIORITIZED). 
 

**All code is written for python 2.7**

### Lots of papers on OCR, and about segmentation of handwritten characters:  
http://iris.usc.edu/Vision-Notes/bibliography/contentschar.html#OCR,%20Document%20Analysis%20and%20Character%20Recognition%20Systems
