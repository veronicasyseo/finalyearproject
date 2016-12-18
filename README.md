# finalyearproject
Hugo, Sigurd, Kevinn and Veronica's Final Year Project @ HKUST 2017
Six Sigma Green Belt project for X Company


Extractandmergelines.py pre-processes images, and feeds them to Tesseract. The OCR output is interpreted. NOTE: in order to use, please create an empty txt file, output.txt in your project folder. Also, download ASN Example.csv from Google Drive. No longer assumes the solid/assort code is below the item code, can also deal with cases where the codes are on the same line.


Charextraction.py finds the locations of the hyphens in the Solid code. Based on this, each character from the Solid code is extracted, and saved on a 28x28 black background. The images can then be fed to tensorflow (or something else) to be read. To use, please first run extractandmergelines.py in order to generate the necessary input file. Future improvements: separation of touching handwritten characters, limiting the number of characters selected to 8 (ab-cde-fgh).


Use benchmark.py to simply use tesseract OCR on any image - will usually not be very accurate

Histograms.py might be useful later. Is related to this paper (about segmenting handwriting): http://www.ee.bgu.ac.il/~dinstein/stip2002/Seminar_papers/David_Cahana_A%20character%20segmentation%20method%20using%20projection%20profile-based%20technique.pdf 

Interpretoutput.py 
Given a txt file containing the OCR output from the initial pre-procesing and an advance shipment notice (ASN) in csv format (see google drive), it categorizes each line of text into the following categories:
Item code, solid (assort) code, quantity, box number, item description, or "nothing". 
This will be useful for determining whether further image processing is needed for a particular input image. 


What needs to be done: 
 - Find data set of segmented (single) letters A-Z and a-z.
 - Find an appropriate way of reading handwritten characters.
 - Implement a way to use histograms for segmentation and cleaning. 
 

All code is written for python 2.7

Lots of papers on OCR, and about segmentation of handwritten characters:  
http://iris.usc.edu/Vision-Notes/bibliography/contentschar.html#OCR,%20Document%20Analysis%20and%20Character%20Recognition%20Systems
