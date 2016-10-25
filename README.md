# finalyearproject
Hugo, Sigurd, Kevinn and Veronica's Final Year Project @ HKUST 2017
Six Sigma Green Belt project for X Company


Extractandmergelines.py addresses some shortcomings of extract_lines.py

ocrandtextanalysis.py starts interpreting the OCR output. NOTE: in order to use ocrandtextanalysis.py, please create an empty txt file, output.txt in your project folder. Also, download ASN Example.csv from Google Drive. If you use images not mentioned in the csv file, you will have to add them if you want to check OCR accuracy. 

Extract_lines.py gives decent outputs

Use benchmark.py to simply use tesseract OCR on any image - will usually not be very accurate


What needs to be done: 
 - Automatic character extraction (single characters) using opencv (could build on extractandmergelines.py.)
 - An idea finding the location of handwritten characters is to use Tesseract to find the location of the first line of the SKU, for example 341-172340(64-24). Then, we can send the text below to another kind of processing, such as CNN, since it is likely to have handwritten text. 
 - Based on ^, save single characters in 28x28 image files, where the characters take up 20x20 and are centered in a 28x28 grid. The purpose of this is to enable us use tensorflow for OCR on handwritten characters.
 - Need a large dataset for training tensorflow (multilayer convolutional neural network). Format should be [img, label], where img is a 28x28 image (see above) and label is the correct value of the character. 
 
 

Lots of papers on OCR, and about segmentation of handwritten characters:  
http://iris.usc.edu/Vision-Notes/bibliography/contentschar.html#OCR,%20Document%20Analysis%20and%20Character%20Recognition%20Systems
