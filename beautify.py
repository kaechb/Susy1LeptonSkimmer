import os


def beautify(file):
    # black is better
    # return os.system("autopep8 --aggressive --in-place {0}".format(file))
    return os.system("black {0}".format(file))


for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".py"):
            print("beautifying:", root + "/" + filename)
            beautify(root + "/" + filename)
