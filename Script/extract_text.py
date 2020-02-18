# -*- coding: utf-8 -*-
from lxml import etree
import re
import codecs
from pathlib import Path
import csv
import sys


def xml_to_text(path):
    tree = etree.parse(path)
    root = tree.getroot()
    en = 0
    fr = 0
    ar = 0
    # Run down in the tree and add to a list the pairs "attribute : value
    T = []
    for document in root:
        for page in document:
            # extract the class of the text
            if page.attrib['Category'] != '':
                # To deal with the documents with multiple pages
                classe = page.attrib['Category']
            # Continue to loop to extract texts
            for zone in page:
                T.append(zone.attrib)
    # From the T list, we will recover the values of
    # the attribute "contents"
    text = []
    for d in T:
        if 'contents' in d:
            text.append(d['contents'])
            if d['language'] == "english":
                en = en + 1
            elif d['language'] == "french":
                fr = fr + 1
            elif d['language'] == "arabic":
                ar = ar + 1
    # Convert our list to a single string
    text_recovered = " ".join(text)
    # Finally let's remove string control characters
    # "\n", "\t" and "\r" using regular expression
    regex = re.compile(r'[\n\r\t]')
    text_recovered = regex.sub(" ", text_recovered)
    # Detect the text's language with a majority vote
    # to handle the files that contain many languages
    vote = max(en, fr, ar)
    if vote == en:
        lang = 'en'
    elif vote == fr:
        lang = 'fr'
    else:
        lang = 'ar'
    return text_recovered, classe, lang


def text_to_file(file_name, text):
    """ Now let's save the text in a txt file """
    # specify the "utf-8" codecs
    file = codecs.open(file_name, "w", "utf-8")
    file.write(text)
    file.close()


def texts_to_files(path_xml, path_txt, name_csv_file):
    """ Now let's iterate over all xml files and extract the text """
    path_list = Path(path_xml).glob('**/*.xml')
    # create a csv file
    headers = ["Filename", "Text", "Class", "Language"]
    csvfile = codecs.open(name_csv_file, 'w', "utf-8")
    filewriter = csv.writer(csvfile, delimiter='\t')
    filewriter.writerow(headers)
    for filename in path_list:
        # convert a path object to a string to avoid parsing errors
        filename = str(filename)
        # replace xml by txt
        string = filename.replace("xml", "txt")
        # replace "\" by "/" to support UNIX systems
        string = string.replace("\\", "/")
        Text, Classe, Lang = xml_to_text(filename)
        filewriter.writerow([string, Text, Classe, Lang])
        text_to_file(string, Text)
    csvfile.close()


# For instance, we can pass arguments through the command line :
# "python extract_text.py ../Data/train2/xml
# ../Data/train2/txt/ ../Data/train2/train_text.csv"


if __name__ == "__main__":
    texts_to_files(sys.argv[1], sys.argv[2], sys.argv[3])
