
import sys
import os.path
import re

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "usage: create_csv <base_path>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=";"

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                labellst = subject_path.split('/')
                labelele = labellst[5]
                labeldig = re.findall(r'\d+', labelele)
                label = int (labeldig[0])


                print "%s%s%d" % (abs_path, SEPARATOR, label)
            
