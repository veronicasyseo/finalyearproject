import re


class outputInterpreter():

    def itemCand(self, line):
        # perform conversions
        line = line.replace(" ", "")
        line = line.replace("b", "6")
        line = line.replace("B", "8")
        line = line.upper()
        line = line.replace("O", "0")
        line = line.replace("I", "1")
        line = line.replace("L", "1")
        print line
        # determine if the line is a candidate for item code or not
        five_or_more_digits_consecutive = re.findall(r"\d{5,20}", line, re.UNICODE)
        expression_xx_xx = re.findall(r"\d{2}-\d{2}", line, re.UNICODE)
        expression_xxx_xxxx = re.findall(r"\d{3}-\d{4,10}", line, re.UNICODE)
        # print five_or_more_digits_consecutive
        # print expression_xx_xx
        # print expression_xxx_xxxx

        digits = 0
        for c in line:
            if c.isdigit():
                digits += 1

        if digits > 8:
            nine_or_more_digits = True
        else:
            nine_or_more_digits = False

        if len(five_or_more_digits_consecutive) > 0 or len(expression_xx_xx) > 0 or len(expression_xxx_xxxx) or nine_or_more_digits:
            return True
        else:
            return False

    def solidCand(self, line):
        # perform conversions
        line = line.replace(" ", "")
        line = line.replace("b", "6")
        line = line.replace("B", "8")
        line = line.upper()
        line = line.replace("O", "0")
        line = line.replace("I", "1")
        line = line.replace("L", "1")
        print line
        solidcode = False
        # determine is it's a candidate for solid code or not

        triple_zeros = re.findall(r"000", line, re.UNICODE)

        double_zeros_and_chars = re.search(r"00", line, re.UNICODE)  # first check if have double 00
        # next, based on the positions for the double zeros, want to check for other chars
        if double_zeros_and_chars:
            d = double_zeros_and_chars.start() > 1
            if d:
                solidcode = True

        format_xxx_hyphen = re.findall(r"-.{3}-", line, re.UNICODE)

        if len(re.findall(r"\d{8}", line, re.UNICODE)) > 0 or len(format_xxx_hyphen) > 0 or len(triple_zeros) > 0:
            solidcode = True

        return solidcode

    def assortCand(self, line):
        # format the string
        line = line.replace(" ", "")
        line = line.upper()
        print line
        # look for formats satisfying the setup of assort code
        c_d_d = re.findall(r"[A-Z]{1}\d{2,4}", line, re.UNICODE)

        if len(c_d_d) > 0 or (0 < len(line) < 5):
            return True
        else:
            return False

    def packCand(self, line):
        # conversion of text
        line = line.replace(" ", "")
        line = line.upper()

        d_d_c_c = re.findall(r"\d{2}[A-Z]{2}", line, re.UNICODE)

        if len(d_d_c_c) > 0 or ("PC" in line or "PIECE" in line or "SIZE" in line):
            return True
        else:
            return False

    def cartonCand(self, line):
        line = line.replace(" ", "")
        line = line.upper()

        if "NO" in line or "N0" in line or "NUM" in line or "CAR" in line or "CTN" in line:
            return True
        else:
            return False

    def descriptionCand(self, line):
        line = line.upper()

        spaces_around = re.findall(r"\s", line, re.UNICODE)

        line = line.replace(" ", "")

        six_or_more_chars = re.findall(r"[A-Z]{6,90}", line, re.UNICODE)

        if len(spaces_around) > 4 or len(six_or_more_chars) > 0:
            return True
        else:
            return False

    def categorizeLines(self, ocr_output):
        """Input: String output from Tesseract
        Output: A dictionary containing the indeces for the useful pieces of information"""

        dict_indeces = {'Itemcode': False, 'Solidcode': False, 'Assortcode': False, 'Pack size': False, 'Text description': False}  # initialize dictionary

        # step 1: split the ocr output to a list of lines
        list_lines = re.split('\n', ocr_output)

        candidate_info = []
        for x in range(0, len(list_lines)):
            candidate_info.append([])  # will store the candidate information in text format, and then use the keyword "in" to decide how to treat the lines later

        # first pass: find the candidate categories for each line
        k = 0
        for line in list_lines:
            if self.itemCand(line):
                print "Item code candidate found"
                candidate_info[k].append("Itemcode")

            if self.solidCand(line):
                print "Solid code candidate found"
                candidate_info[k].append("Solidcode")

            if self.assortCand(line):
                print "Assort code candidate found"
                candidate_info[k].append("Assortcode")

            if self.packCand(line):
                print "Pack size candidate found"
                candidate_info[k].append("Pack size")

            if self.cartonCand(line):
                print "Carton number candidate found"
                candidate_info[k].append("Carton number")

            if self.descriptionCand(line):
                print "Text description candidate found"
                candidate_info[k].append("Text description")

            k += 1
        print candidate_info

        # Start the second pass: Uniquely determining the indeces for each category

        min_index_dict = {'Itemcode': 0, 'Solidcode': 1, 'Assortcode': 1,
                          'Pack size': 2, 'Carton number': 3, 'Text description': 4}
        max_index_dict = {'Itemcode': 100, 'Solidcode': 100, 'Assortcode': 100, 
                          'Pack size': 100, 'Carton number': 100, 'Text description': 100}
        # preparation stage: pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # Scenario 1: easy case
        pack_occur = 0
        pack_index = []
        j = 0
        for line in candidate_info:
            if "Pack size" in line:
                pack_occur += 1
                pack_index.append(j)
            j += 1

        if pack_occur == 1:  # look for the index of carton no
            dict_indeces['Pack size'] = pack_index[0]
            max_index_dict['Itemcode'] = pack_index[0]-1
            max_index_dict['Solidcode'] = pack_index[0]-1
            max_index_dict['Assortcode'] = pack_index[0]-1
            min_index_dict['Text description'] = max(min_index_dict['Text description'], pack_index[0]+2)

        carton_index = []
        j = 0
        for line in candidate_info:
            if "Carton number" in line:
                carton_index.append(j)
            j += 1

        if len(carton_index) == 1:
            max_index_dict['Itemcode'] = min(carton_index[0]-1, max_index_dict['Itemcode'])
            max_index_dict['Solidcode'] = min(carton_index[0]-1, max_index_dict['Solidcode'])
            max_index_dict['Assortcode'] = min(carton_index[0]-1, max_index_dict['Assortcode'])
            min_index_dict['Text description'] = max(min_index_dict['Text description'], carton_index[0]+1)

        # again, pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if min_index_dict[element] > x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # pop all the impossible cases, assuming no lines are omitted in image processing
        for x in range(0, len(candidate_info)):
            j = 0
            for element in candidate_info[x]:
                if max_index_dict[element] < x:
                    candidate_info[x].pop(j)
                j += 1
        print "After removing impossible cases, the new list is: "
        print candidate_info

        # find the index of the text description line if there's a clear-cut case
        descr_count = 0
        descr_index = []
        j = 0
        for line in candidate_info:
            if "Text description" in line:
                descr_count += 1
                descr_index.append(j)
            j += 1

        if descr_count == 1:
            dict_indeces['Text description'] = descr_index[0]
            max_index_dict['Itemcode'] = min(max_index_dict['Itemcode'], descr_index[0]-4)
            max_index_dict['Solidcode'] = min(max_index_dict['Solidcode'], descr_index[0]-3)
            max_index_dict['Assortcode'] = min(max_index_dict['Assortcode'], descr_index[0]-3)
            max_index_dict['Pack size'] = min(max_index_dict['Pack size'], descr_index[0]-2)
            max_index_dict['Carton number'] = min(max_index_dict['Carton number'], descr_index[0]-1)
            max_index_dict['Text description'] = descr_index[0]
        # could do another looping through at this pt to eliminate impossible cases

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

        solid_count = 0
        solid_index = []
        j = 0
        for line in candidate_info:
            if "Solidcode" in line:
                solid_count += 1
                solid_index.append(j)
            j += 1

        if solid_count == 1:
            dict_indeces['Solidcode'] = solid_index[0]
            max_index_dict['Itemcode'] = min(solid_index[0]-1, max_index_dict['Itemcode'])
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], solid_index[0]+1)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], solid_index[0]+2)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], solid_index[0]+3)

            # again, pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if min_index_dict[element] > x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

            # pop all the impossible cases, assuming no lines are omitted in image processing
            for x in range(0, len(candidate_info)):
                j = 0
                for element in candidate_info[x]:
                    if max_index_dict[element] < x:
                        candidate_info[x].pop(j)
                    j += 1
            print "After removing impossible cases, the new list is: "
            print candidate_info

        item_count = 0
        item_index = []
        j = 0

        for line in candidate_info:
            if "Itemcode" in line:
                item_count += 1
                item_index.append(j)
            j += 1

        if item_count == 1:
            dict_indeces['Itemcode'] = item_index[0]
            min_index_dict['Solidcode'] = max(min_index_dict['Solidcode'], item_index[0]+1)
            min_index_dict['Assortcode'] = max(min_index_dict['Assortcode'], item_index[0]+1)
            min_index_dict['Pack size'] = max(min_index_dict['Pack size'], item_index[0]+2)
            min_index_dict['Carton number'] = max(min_index_dict['Carton number'], item_index[0]+3)
            min_index_dict['Text description'] = max(min_index_dict['Text description'], item_index[0]+4)

        # based on min_index_dict and max_index_dict, infer the indeces of the unassigned lines
        for key, value in dict_indeces.iteritems():
            if not value:
                if min_index_dict[key] == max_index_dict[key]:
                    dict_indeces[key] = min_index_dict[key]
        # small issue: get both item and solid code this way, need a way to determine which of the two we have. 
        # Can later use similar methods to previous in order to distinguish between the two categories
        
        print dict_indeces
        print candidate_info
        return False  # to be changed


if __name__ == "__main__":
    interpreter = outputInterpreter()
    # ocr_output = "371-182631(72-12)\n66-003-000\n24 PCS\nNo. 121\nFull Zip Hoodie Large\nF003".encode('utf-8')
    ocr_output = "341-1983160142)\ngyms-000\n24 PCS\nN0. 110\nU LI S Inchgo sweat pullover hoodlc"
    dict_indeces = interpreter.categorizeLines(ocr_output)
