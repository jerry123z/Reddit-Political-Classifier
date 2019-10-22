import sys
import argparse
import os
import json
import html
import re
import spacy

# TODO: fix directory
with open('C:\\Users\\Jerry\\Documents\\CSC401\\A1\\wordlists\\Stopwords') as f:
    stopwords = f.read().splitlines()
    stopwords = list(map(lambda a:r"\s" + a + r"/\S+", stopwords))

with open('C:\\Users\\Jerry\\Documents\\CSC401\\A1\\wordlists\\abbrev.english') as f:
    abbrev = f.read().splitlines()

indir = 'C:\\Users\\Jerry\\Documents\\CSC401\\A1\\data';

nlp = spacy.load('en', disable=['parser', 'ner'])

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment
    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = comment
    if 1 in steps:
        # Remove newline characters
        modComm = re.sub(re.compile("\n"), " ", modComm)
    if 2 in steps:
        "remove html escape characters"
        modComm = html.unescape(modComm)
    if 3 in steps:
        "remove hyperlinks"
        http_pattern = re.compile(r"http\S+(\s)?")
        modComm = re.sub(http_pattern, "", modComm)
        www_pattern = re.compile(r"www.\S+(\s)?")
        modComm = re.sub(www_pattern, "", modComm)
    if 4 in steps:
        "remove punctuation"
        punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
        punctuation_pattern = re.compile("([" + punctuation + r"])+(\s)*?")
        modComm = re.sub(punctuation_pattern, " "+ r'\1' + " ", modComm)
    if 5 in steps:
        # fix clitics
        modComm = re.sub("'", " '", modComm)
        modComm = re.sub("n 't", " n't", modComm)
    if 6 in steps:
        # POS Tag each word using spacy
        word_list = []
        doc = nlp(modComm)
        for token in doc:
            word_list.append(token.text + "/" + token.tag_)
        modComm = " ".join(word_list)
    if 7 in steps:
        # remove stopwords
        modComm = re.sub("|".join(stopwords), "", modComm, flags=re.IGNORECASE)
        modComm = " ".join(modComm.split())
    if 8 in steps:
        # use spacy to apply lemmatization
        token_list = modComm.split(" ")
        token_list = list(filter(lambda a: a != '', token_list))
        word_list = []
        for i in range(len(token_list)):
            word = re.sub(r'(\W*)/\S+', r'\1', token_list[i])
            word_list.append(word)

        comment = " ".join(word_list)
        doc = nlp(comment)
        output_list = []
        for token in doc:
            output_list.append(token.lemma_ + "/" + token.tag_)
        modComm = " ".join(output_list)
    if 9 in steps:
        # add newline between each sentence
        token_list = modComm.split(" ")
        token_list = list(filter(lambda a: a != '', token_list))
        if len(token_list) != 0:
            prev_word = token_list[0]
            for i in range(len(token_list)):
                word = re.sub(r'(\W*)/\S+', r'\1', token_list[i])
                pos = token_list[i].replace(word, "", 1)
                if pos == "/.":
                    if not (prev_word + word in abbrev):
                        token_list[i] = token_list[i] + " \n"
                prev_word = word
            modComm = " ".join(token_list)

    if 10 in steps:
        # change everything to lower case
        token_list = modComm.split(" ")
        token_list = list(filter(lambda a: a != '', token_list))
        for i in range(len(token_list)):
            word = re.sub(r'(\W*)/\S+', r'\1', token_list[i])
            pos = token_list[i].replace(word, "", 1)
            token_list[i] = word.lower() + pos
        modComm = " ".join(token_list)

    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile, encoding='UTF8'))
            max = int(args.ID[0]) % len(data) + int(args.max)
            circular = 0
            if max >= len(data):
                circular = max - len(data)
                max = len(data)

            for line in data[int(args.ID[0]) % len(data): max]:
                j = json.loads(line)
                text = j["body"]
                data = {}
                data["id"] = j["id"]
                data["cat"] = file
                output = preproc1(text)
                data["body"] = output
                json.dumps(data)
                allOutput.append(data)

            if circular != 0:
                for line in data[0: circular]:
                    j = json.loads(line)
                    text = j["body"]
                    data = {}
                    data["id"] = j["id"]
                    data["cat"] = file
                    output = preproc1(text)
                    data["body"] = output
                    json.dumps(data)
                    allOutput.append(data)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
