""""coding: UTF-8"""
import csv
import Levenshtein

# Keep in mind that strings are immutable in Python
# May want to introduce a fake line in the csv file (ASN) for error detection.
# Objective: categorize each line.

ocr_input_path = "output.txt" # path to your OCR output 
asn_input_path = "path to advance shipment notice in csv format"

ocr_data = []
asn_data = []

with open(asn_input_path, "rb") as csvfile: # load the ASN from csv file, store in array (line by line)
    spamreader = csv.reader(csvfile, delimiter = ';', quotechar = '|') # Spamreader is an object..need to save the info somehow

    for row in spamreader: # Row only contains 1 row at any given time
        print ', '.join(row)
        asn_data.append(row)
csvfile.close()

with open(ocr_input_path) as f: # load the OCR output from a txt file, place in an array.
    ocr_data.append(f.readlines()) # attaches everything to the 0th element of ocr_data.
f.close()


def inside(st):

    """Check if the string has 6 or more digits in a row somewhere. Return 1 if yes."""
    contains = False
    for x in range(8, len(st)):
        if st[(x-8):x].isdigit():
            contains = True
    return contains


def isitemcode(input_string):
    "Returns 1 if the string is likely to be an item code."
    "Returns 0 if it's not likely. "
    input_string = input_string.strip('\n') # get rid of the end-line character
    input_string = input_string.strip(" ")
    input_string = input_string.upper()
    input_string = input_string.replace("O", "0")
    input_string = input_string.replace("L", "1")
    input_string = input_string.translate(None, '-_() ')
    if input_string.isdigit(): # should not rely on this, since there many be noise at the end of the string
        return 1
    elif inside(input_string):
        return 1
    else:
        return 0


def isquantity(st):

    """Returns 1 if it's likely that this line is telling the quantity of the parcel"""
    st = st.upper()

    if "PC" in st or "PCS" in st or "PIECES" in st:
        return 1
    else:
        return 0


def isdescription(st): # based on counting the number of whitespaces found in the string. Potential issue: Could be as few as 1 whitespace...
    # may or may not want to convert zeros to ohs
    """Returns 1 if it's likely that this is a text description of the parcel contents"""
    # alternatively, look at the length, and check if contains characters only? cannot do, since contains some special characters
    st = st.upper()
    st = st.replace("0", "O")
    st = st.translate(None, "-./| ")

    if st.isalpha() and len(st) > 5: # 5 selected arbitrarily..
        return 1
    else:
        return 0


def isboxnumber(st):

    """Returns 1 if it's likely that this is just the box number (useless information)"""
    st = st.upper()
    if "NO" in st or "N0" in st: # parameter for calibration
        return 1
    else:
        return 0


def issolidcode(st):

    """Returns 1 if it's likely that this is a solid code.
    Problem: Both item and solid codes can contain two hyphens. """

    if st.count("-") == 2 or "000" in st:
        return 1
    else:
        return 0


# Want to do basic filtering: Is it rubbish or not? - if there are many lines (>10), then expect it to be rubbish
if len(ocr_data[0]) > 20:
    print "Need to use other image processing techniques! The current one only gives gibberish!"
else: # want to categorize the different lines into types (description, code etc)
    # may want to keep an array to check if there's any confusion about the characteristics of the line
    di_array = []
    # create an array with an element for each line in the txt file, to keep track of the category
    cats = []  # based on the original position of the lines, including blank lines, for the purpose of keeping location information.

    for line in ocr_data[0]:
        di = {'Itemcode': False, 'Solidcode': False, 'Quantity': False, 'Boxnumber': False, 'Description': False, 'Itemcodeminlev': False, 'Solidcodeminlev': False, 'Nothing': False, 'Descriptionratio': False, 'Categorized': False}
        line = line.strip("\n") # remove the new-line character

        if len(line) > 0: # if a line is just empty, we don't want to consider it
            if isitemcode(line) == 1:
                # find the best fitting item code and the corresponding lev dist
                smallest_lev = 888
                smallest_pos = 888

                for i in range(1, len(asn_data)):
                    d = Levenshtein.distance(asn_data[i][3], line)
                    if d < smallest_lev:
                        smallest_lev = d
                        smallest_pos = i
                di['Itemcode'] = asn_data[smallest_pos][3]
                print "Best fit is: " + str(asn_data[smallest_pos][3])
                print "The Lev. distance is: " + str(smallest_lev)
                di['Itemcodeminlev'] = smallest_lev

            if isquantity(line) == 1:
                di['Quantity'] = True

            if isboxnumber(line) == 1:
                di['Boxnumber'] = True

            descr_lev = 888
            descr_index = 888

            if isdescription(line) == 1:
                for i in range(1, len(asn_data)):
                    d = Levenshtein.distance(asn_data[i][2], line)
                    if d < descr_lev:
                        descr_lev = d
                        descr_index = i
                di['Description'] = asn_data[descr_index][2]
                di['Descriptionratio'] = 1.0*descr_lev/len(line)

            solid_lev = 888
            solid_index = 888

            if issolidcode(line) == 1: # should look through solid codes to check if there's any match (but may not be accurate to do this
                for i in range(1, len(asn_data)):
                    d = Levenshtein.distance(asn_data[i][4], line)
                    if d < solid_lev:
                        solid_lev = d
                        solid_index = i
                di['Solidcode'] = asn_data[solid_index][4]
                di['Solidcodeminlev'] = solid_index
                print "The best fitting solid code was: " + str(asn_data[solid_index][4])
                di['Solidcode'] = True

            # next, need to consider which it's most likely to be...but don't yet have the information from the others..

            print di
            di_array.append(di)
            # based on the above, want to narrow down what the line is.
        else:
            di['Nothing'] = True
            di_array.append(di) # to maintain positions

        cats.append(False) # To be amended, need to keep position information.

    # end of the for-loop, now look through each of the elements.

    # find the smallest lev distance for the item code.
    min_dist = 888
    min_dist_index = 0
    for x in range(0, len(di_array)): # Item code part
        if not di_array[x]['Itemcode'] == False:
            if di_array[x]['Itemcodeminlev'] < min_dist:
                min_dist = di_array[x]['Itemcodeminlev']
                min_dist_index = x
    cats[min_dist_index] = "Itemcode"
    di_array[min_dist_index]['Categorized'] = True

    min_ratio = 888
    min_ratio_index = 888

    for x in range(0, len(di_array)): # Find the text description of the parcel
        if not di_array[x]['Categorized']:
            if di_array[x]['Descriptionratio'] < min_ratio and di_array[x]['Descriptionratio'] is not False: # Check if a 0 would be evaluated as False
                min_ratio = di_array[x]['Descriptionratio']
                min_ratio_index = x
    print min_ratio
    cats[min_ratio_index] = "Description"
    di_array[min_ratio_index]['Categorized'] = True

    # for x in range(0, len(di_array)): # check if it's the quantity -- how to compare? See if a substring is numeric?

    for x in range(0, len(di_array)):
        if di_array[x]['Nothing']:
            cats[x] = "Nothing"
            di_array[x]['Categorized'] = True

    # all uncategorized lines before Itemcode line: Turn into Nothing.
    # Step 1: Find the index of the Itemcode...min_dist_index
    for x in range(0, min_dist_index):
        if not di_array[x]['Categorized']:
            di_array[x]['Nothing'] = True
            di_array[x]['Categorized'] = True
            cats[x] = 'Nothing'

    # check if there are 3 Falses left (uncategorized lines), if yes, it'll be easier to categorize those last lines.

    count_false = 0

    for x in cats:
        if x == False:
            count_false += 1

    if count_false == 3: # hope for convenient structure
        count_boxnum = 0
        count_quant = 0
        count_solid = 0

        for ele in di_array:
            if not ele['Categorized']:
                if ele['Boxnumber']:
                    count_boxnum += 1
                if ele['Quantity']:
                    count_quant += 1
                if ele['Solidcode']:
                    count_solid += 1
            if count_boxnum == count_quant == count_solid == 1: # ez m8
                for i in range(0, len(di_array)): # may need to use a while-loop instead
                    if di_array[i]['Categorized'] == False: # want to count the number of falses for this element
                        if di_array[i]['Boxnumber'] == True and di_array[i]['Quantity'] == False and di_array[i]['Solidcode'] == False:
                            di_array[i]['Categorized'] = True
                            cats[i] = 'Boxnumber'
                        elif di_array[i]['Boxnumber'] == False and di_array[i]['Quantity'] == True and di_array[i]['Solidcode'] == False:
                            di_array[i]['Categorized'] = True
                            cats[i] = 'Quantity'
                        elif di_array[i]['Boxnumber'] == False and di_array[i]['Quantity'] == False and di_array[i]['Solidcode'] == True:
                            di_array[i]['Categorized'] = True
                            cats[i] = 'Solidcode'

    # Solid code part should be left for a later part...
    print cats
