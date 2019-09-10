import numpy as np
import sys
import argparse
import os
import json
import re
import pandas as pd

# TODO fix directories
bn = pd.read_csv(r"C:\Users\Jerry\Documents\CSC401\A1\wordlists\BristolNorms+GilhoolyLogie.csv")
war = pd.read_csv(r"C:\Users\Jerry\Documents\CSC401\A1\wordlists\Ratings_Warriner_et_al.csv")

with open(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Alt_IDs.txt") as f:
    alt_ids = f.read().splitlines()
with open(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Right_IDs.txt") as f:
    right_ids = f.read().splitlines()
with open(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Center_IDs.txt") as f:
    center_ids = f.read().splitlines()
with open(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Left_IDs.txt") as f:
    left_ids = f.read().splitlines()
alt_data = np.load(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Alt_feats.dat.npy")
right_data = np.load(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Right_feats.dat.npy")
center_data = np.load(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Center_feats.dat.npy")
left_data = np.load(r"C:\Users\Jerry\Documents\CSC401\A1\feats\Left_feats.dat.npy")


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros(173+1)
    words = get_words(comment["body"])
    tags = get_tags(comment["body"])
    feats[0] = first_person_pronouns(words)
    feats[1] = second_person_pronouns(words)
    feats[2] = third_person_pronouns(words)
    feats[3] = coordinating_conjunctions(tags)
    feats[4] = past_tense_verb(tags)
    feats[5] = future_tense_verb(words, tags)
    feats[6] = commas(words)
    feats[7] = punctuation(words)
    feats[8] = common_nouns(tags)
    feats[9] = proper_nouns(tags)
    feats[10] = adverbs(tags)
    feats[11] = wh(tags)
    feats[12] = acronym(words)
    feats[13] = 0
    feats[14] = average_sentence_length(tags)
    feats[15] = average_word_length(words, tags)
    feats[16] = number_of_sentences(tags)
    list_aoa = aoa_list(words)
    list_img = img_list(words)
    list_fam = fam_list(words)
    feats[17] = np.mean(list_aoa)
    feats[18] = np.mean(list_img)
    feats[19] = np.mean(list_fam)
    feats[20] = np.std(list_aoa)
    feats[21] = np.std(list_img)
    feats[22] = np.std(list_fam)
    list_v_mean_sum = v_mean_sum_list(words)
    list_a_mean_sum = a_mean_sum_list(words)
    list_d_mean_sum = d_mean_sum_list(words)
    feats[23] = np.mean(list_v_mean_sum)
    feats[24] = np.mean(list_a_mean_sum)
    feats[25] = np.mean(list_d_mean_sum)
    feats[26] = np.std(list_v_mean_sum)
    feats[27] = np.std(list_a_mean_sum)
    feats[28] = np.std(list_d_mean_sum)
    feats[29:173] = receptiviti(comment)
    return feats


def get_words(comment):
    comment = comment.replace("\n ", "")
    tokens = comment.strip().split(' ')
    tokens = list(map(lambda a: a.split("/")[0], tokens))
    return tokens


def get_tags(comment):
    comment = comment.replace("\n ", "")
    tokens = comment.strip().split(' ')
    # bandaid fix for bad Part 1
    if tokens == [""]:
        tokens = ["/"]
    tokens = list(map(lambda a: a.split("/")[-1], tokens))
    return tokens


def first_person_pronouns(words):
    pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    count = 0
    for word in words:
        if word in pronouns:
            count += 1
    return count


def second_person_pronouns(words):
    pronouns = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    count = 0
    for word in words:
        if word in pronouns:
            count += 1
    return count


def third_person_pronouns(words):
    pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
    count = 0
    for word in words:
        if word in pronouns:
            count += 1
    return count


def coordinating_conjunctions(tags):
    count = 0
    for tag in tags:
        if tag == "CC":
            count += 1
    return count


def past_tense_verb(tags):
    count = 0
    for tag in tags:
        if tag == "VBD":
            count += 1
    return count


def future_tense_verb(words, tags):
    future_words = ["'ll", 'will', 'gonna']
    count = 0
    for word in words:
        if word in future_words:
            count += 1

    # now for 'going to VB'
    if len(words) > 2:
        for i in range(2, len(words)):
            if words[i-2] == "going" and words[i-1] == "to" and tags[i] == "VB":
                count += 1
    return count


def commas(words):
    count = 0
    for word in words:
        if word == ",":
            count += 1
    return count


def punctuation(words):
    count = 0
    non_punc = re.compile(r"\W+")
    for word in words:
        if re.sub(non_punc, '', word) == "" and len(word) > 1:
            count += 1

    return count


def common_nouns(tags):
    nouns = ['NN', 'NNS']
    count = 0
    for tag in tags:
        if tag in nouns:
            count += 1
    return count


def proper_nouns(tags):
    nouns = ['NNP', 'NNPS']
    count = 0
    for tag in tags:
        if tag in nouns:
            count += 1
    return count


def adverbs(tags):
    adverbs_tags = ['RB', 'RBR', 'RBS']
    count = 0
    for tag in tags:
        if tag in adverbs_tags:
            count += 1
    return count


def wh(tags):
    wh_tags = ['WDT', 'WP', 'WP$', 'WRB']
    count = 0
    for tag in tags:
        if tag in wh_tags:
            count += 1
    return count

def acronym(words):
    slang = ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao',
             'sml', 'btw', 'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
             'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc',
             'tmi', 'ym', 'ur', 'u', 'sol', 'fml']
    count = 0
    for word in words:
        if word in slang:
            count += 1
    return count


def average_sentence_length(tags):
    unsplit_sentences = " ".join(tags)
    split_sentences = unsplit_sentences.split(".")
    return float(len(unsplit_sentences) - len(split_sentences))/len(split_sentences)


def average_word_length(words, tags):
    punctuation_tags = ['#', '$', '.', ',', ':', '(', ')', '"']
    word_lengths = []
    for i in range(len(words)):
        if tags[i] not in punctuation_tags:
            word_lengths.append(len(words[i]))
    result = 0
    if len(word_lengths) != 0:
        result = float(sum(word_lengths)) / len(word_lengths)
    return result


def number_of_sentences(tags):
    return tags.count(".")


# All the lists are the same, use Pandas to find the corresponding value to the word and append it
def aoa_list(words):
    value_list = []
    for word in words:
        if bn["AoA (100-700)"][bn["WORD"] == word].data:
            value_list.append(bn["AoA (100-700)"][bn["WORD"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def img_list(words):
    value_list = []
    for word in words:
        if bn["IMG"][bn["WORD"] == word].data:
            value_list.append(bn["IMG"][bn["WORD"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def fam_list(words):
    value_list = []
    for word in words:
        if bn["FAM"][bn["WORD"] == word].data:
            value_list.append(bn["FAM"][bn["WORD"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def v_mean_sum_list(words):
    value_list = []
    for word in words:
        if war["V.Mean.Sum"][war["Word"] == word].data:
            value_list.append(war["V.Mean.Sum"][war["Word"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def a_mean_sum_list(words):
    value_list = []
    for word in words:
        if war["A.Mean.Sum"][war["Word"] == word].data:
            value_list.append(war["A.Mean.Sum"][war["Word"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def d_mean_sum_list(words):
    value_list = []
    for word in words:
        if war["D.Mean.Sum"][war["Word"] == word].data:
            value_list.append(war["D.Mean.Sum"][war["Word"] == word].mean())
        else:
            value_list.append(0)
    return value_list


def receptiviti(comment):
    # just grab catagory, then index, then the data
    if comment["cat"] == "Alt":
        index = alt_ids.index(comment['id'])
        output = alt_data[index]
    elif comment["cat"] == "Right":
        index = right_ids.index(comment['id'])
        output = right_data[index]
    elif comment["cat"] == "Center":
        index = center_ids.index(comment['id'])
        output = center_data[index]
    else:
        index = left_ids.index(comment['id'])
        output = left_data[index]
    return output


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    for i in range(len(feats)):
        print(i)
        # Bandaid fix for having empty comments come out of preproc
        if data[i]["body"] == "":
            continue
        feats[i] = extract1(data[i])
        if data[i]["cat"] == "Left":
            feats[i][173] = 0
        elif data[i]["cat"] == "Center":
            feats[i][173] = 1
        elif data[i]["cat"] == "Right":
            feats[i][173] = 2
        else:
            feats[i][173] = 3
    np.savez_compressed(args.output, feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
