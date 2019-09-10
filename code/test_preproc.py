import unittest
import html
import re

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
        modComm = re.sub(re.compile("\n"), "", modComm)
    if 2 in steps:
        modComm = html.unescape(modComm)
    if 3 in steps:
        http_pattern = re.compile("http\S+(\s)?")
        modComm = re.sub(http_pattern, "", modComm)
        www_pattern = re.compile("www.\S+(\s)?")
        modComm = re.sub(www_pattern, "", modComm)
    if 4 in steps:
        punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
        punctuation_pattern = re.compile("([" + punctuation + "])+(\s)*?")
        modComm = re.sub(punctuation_pattern, " "+ r'\1' + " ", modComm)
    if 5 in steps:
        modComm = re.sub("'", " '", modComm)
        modComm = re.sub("n 't", " n't", modComm)
    if 6 in steps:
        print('TODO')
    if 7 in steps:
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

    return modComm


class TestStringMethods(unittest.TestCase):

    def test_rm_newlines(self):
        s = "\nhello.\nworld\n\nThis is a cool story bro.\n"
        self.assertEqual("hello.worldThis is a cool story bro.", preproc1(s))

    def test_html_chars_to_ascii(self):
        s = "Hello&#33"
        self.assertEqual("Hello!", preproc1(s))

        s = "Hello&Goodbye!"
        self.assertEqual("Hello&Goodbye!", preproc1(s))

    def test_rm_urls(self):
        # TODO: ADD EDGE CASES
        s = "Hello. http://123.456!"
        self.assertEqual("Hello. ", preproc1(s))

        s = "Hello. http://123.456! how are you"
        self.assertEqual("Hello.  how are you", preproc1(s))

        s = "Hello. (http://123.456!) how are you"
        #self.assertEqual("Hello. ( how are you", preproc1(s))

        s = "Hello. (www.123.456.org/uk?123=4567!) how are you"
        #self.assertEqual("Hello. ( how are you", preproc1(s))

    def test_split_punc(self):
        s1 = "Hello. How are you?"
        exp1 = "Hello . How are you ? "

        s2 = "Hello.How are (\"you\")."
        exp2 = "Hello . How are (\" you \"). "

        s3 = "Hi!?! Hi!?!"
        exp3 = "Hi !?! Hi !?! "

        s4 = "Hello.How are, I.e. I.E. e.g., (\"you\")."
        exp4 = "Hello . How are , I.e. I.E. e.g. , (\" you \"). "

        s5 = "Hello. How are you?? I can't believe I saw you!!!"
        exp5 = "Hello . How are you ?? I can't believe I saw you !!!"

        self.assertEqual(exp1, preproc1(s1))
        self.assertEqual(exp2, preproc1(s2))
        self.assertEqual(exp3, preproc1(s3))
        self.assertEqual(exp4, preproc1(s4))

    def test_split_clitic(self):
        s1 = "Can't you shut up about dinosaurs' paws"
        exp1 = "Ca n't you shut up about dinosaurs ' paws"

        self.assertEqual(exp1, preproc1(s1))

    # def test_lemmatize(self):
    #     # lemmatize("Hello . How are you ?")
    #     pass

    def test_sentence_new_line(self):
        s1 = "Hello . \" How are you ?! \" How are you ?!\n We hope you had " \
             "a nice day because I sure didn't . The teddy bear is a Dr. Ford ."
        exp1 = "Hello .\n\" How are you ?!\n\" How are you ?!\n" \
               "We hope you had a nice day because I sure didn't .\n The " \
               "teddy bear is a Dr. Ford .\n"
        self.assertEqual(exp1, sentence_new_line(s1))


if __name__ == '__main__':
    unittest.main()
