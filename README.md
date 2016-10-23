# finalyearproject
Hugo, Sigurd, Kevinn and Veronica's Final Year Project @ HKUST 2017
Six Sigma Green Belt project for X Company


Extractandmergelines.py addresses some shortcomings of extract_lines.py

Extract_lines.py gives decent outputs

Use benchmark.py to simply use tesseract OCR on any image - will usually not be very accurate


What needs to be done: 
 - Automatic character extraction (single characters) using opencv (could build on extractandmergelines.py.)
 - Based on ^, save single characters in 28x28 image files, where the characters take up 20x20 and are centered in a 28x28 grid. The purpose of this is to enable us use tensorflow for OCR on handwritten characters.
 - Need a large dataset for training tensorflow (multilayer convolutional neural network). Format should be [img, label], where img is a 28x28 image (see above) and label is the correct value of the character. 
 
 
