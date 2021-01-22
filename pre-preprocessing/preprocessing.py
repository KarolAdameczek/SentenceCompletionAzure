import gzip
import os, re, pathlib, sys
from xml.etree import ElementTree as ET

class TreeBuilderWithComments(ET.TreeBuilder):
    def comment(self, data):
        self.start("comment", {})
        self.data(data)
        self.end("comment")

PUNCTUATION_DIRECTORY = "punctiuation/"
NO_PUNCTUATION_DIRECTORY = "nopunctiuation/"
BYTE_LIMIT = 1024 #1024 * 1024 * 512  # 512MB


file_counter = 1
file_with_punctuation = open(PUNCTUATION_DIRECTORY + str(file_counter) + ".txt", 'w', encoding='utf8')
file_without_punctuation = open(NO_PUNCTUATION_DIRECTORY + str(file_counter) + ".txt", 'w', encoding='utf8')
    

def main():
    filepaths = getFilePaths()

    for path in progressbar(filepaths, "Processed files: "):
        root = openXMLFile(path)
        parseFile(root)

    file_with_punctuation.close()
    file_without_punctuation.close()

def parseFile(root):
    for s in root.iter(str(root.tag[:-9] + 's')):
        check_file_sizes()
        first_p = True
        first_non_p = True
        for comment in s.iter("comment"):
            text = comment.text.strip().lower()
            if first_p:
                first_p = False
                file_with_punctuation.write(text)
            else:
                file_with_punctuation.write(" " + text)

            if(re.match(r"^[a-zA-ZżźćńółęąśŻŹĆĄŚĘŁÓŃ\d]+$", text)):
                if first_non_p:
                    first_non_p = False
                    file_without_punctuation.write(text)
                else:
                    file_without_punctuation.write(" " + text)

        file_with_punctuation.write("\n")
        file_without_punctuation.write("\n")


def openXMLFile(filename):
    xml_file = gzip.open(filename, mode='r')
    comment_handler = TreeBuilderWithComments()
    parser = ET.XMLParser(target= comment_handler)
    return ET.parse(xml_file, parser=parser).getroot()

def getFilePaths():
    return list(pathlib.Path(os.path.dirname(__file__)).glob('**/ann_segmentation.xml.gz'))

def check_file_sizes():
    if file_with_punctuation.tell() > BYTE_LIMIT:
        next_files()

def next_files():
    global file_with_punctuation
    global file_without_punctuation
    global file_counter

    file_with_punctuation.close()
    file_without_punctuation.close()

    file_counter += 1

    file_with_punctuation = open(PUNCTUATION_DIRECTORY + str(file_counter) + ".txt", 'w', encoding='utf8')
    file_without_punctuation = open(NO_PUNCTUATION_DIRECTORY + str(file_counter) + ".txt", 'w', encoding='utf8')


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "█"*x, " "*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

if __name__ == "__main__":
    main()