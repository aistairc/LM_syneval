import argparse
import pickle
import os
import subprocess
import operator
import logging
from progress.bar import Bar
from tester.TestWriter import TestWriter
from template.TestCases import TestCase

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Parameters for testing a language model")

parser.add_argument('--template_dir', type=str, default='../EMNLP2018/templates',
                    help='Location of the template files')
parser.add_argument('--output_file', type=str, default='all_test_sents.txt',
                    help='File to store all of the sentences that will be tested')
parser.add_argument('--model', type=str, default='../models/model.pt',
                    help='The model to test')
parser.add_argument('--lm_data', type=str, default='../models/model.bin',
                    help='The model .bin file that accompanies the model (for faster loading)')
parser.add_argument('--tests', type=str, default='all',
                    help='Which constructions to test (agrmt/npi/all)')
parser.add_argument('--model_type', type=str, required=True,
                    help='Which kind of model (RNN/multitask/ngram/myRNN)')
parser.add_argument('--unit_type', type=str, default='word',
                    help='Kinds of units used on language model (word/char)')
parser.add_argument('--ngram_order', type=int, default=5,
                    help='Order of the ngram model')
parser.add_argument('--vocab', type=str, default='ngram_vocab.pkl',
                    help='File containing the ngram vocab')
parser.add_argument('--myrnn_dir', type=str, help='Path to lm directory for my rnn')
parser.add_argument('--lm_output', type=str, default='lm_output',
                    help='Path to directory where result files are saved')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--capitalize', action='store_true')

args = parser.parse_args()

def collect_tests(args):
    files = [f[:-7] for f in os.listdir(args.template_dir) if f.endswith('.pickle')]
    if args.tests == 'argmt':
        return [f for f in files if f.find('npi') == -1]
    elif args.tests == 'npi':
        return [f for f in files if f.find('npi') != -1]
    else:
        return files

def mk_all_test_sents():
    tests = collect_tests(args)

    all_test_sents = {}
    for test_name in tests:
        test_sents = pickle.load(open(args.template_dir+"/"+test_name+".pickle", 'rb'))
        all_test_sents[test_name] = test_sents

    return all_test_sents

def write_tests(all_test_sents):
    """There are two aims for this function:

    1) fill the output file with sentences specified by args.output_file;
    2) return two intermediate objects (name_lengths and key_lengths) which will be created
       duing the first step.

    The first step is not essential for the later process; outputing actual sentences are
    rather for interpreation of the results. The intermediate files are more important.

    For this reason, as well as for allowing parallel processing, when the file already exits,
    we avoid to overwrite it, and just create the two intermediate objects.
    """
    writer = TestWriter(args.template_dir, args.output_file)
    out_fn = writer.out_file
    if os.path.exists(out_fn):
        writer.only_fill_maps(all_test_sents)
    else:
        writer.write_tests(all_test_sents, args.unit_type)
    return writer.name_lengths, writer.key_lengths

all_test_sents = mk_all_test_sents()
name_lengths, key_lengths = write_tests(all_test_sents)

def test_LM():
    if not os.path.exists(args.lm_output):
        os.makedirs(args.lm_output)
    lm_output_path = os.path.join(args.lm_output, 'scores.txt')
    results_path = os.path.join(args.lm_output, 'results.pickle')

    if args.model_type.lower() == "ngram":
        logging.info("Testing ngram...")
        os.system('ngram -order ' + str(args.ngram_order) + ' -lm ' + args.model + ' -vocab ' + args.vocab + ' -ppl ' + args.template_dir+'/'+args.output_file + ' -debug 2 > ' + lm_output_path)
        if args.ngram_order == 1:
            results = score_unigram()
        else:
            results = score_ngram()
    elif args.model_type.lower() == 'myrnn':
        logging.info("Testing My RNN...")
        test_path = os.path.join(args.template_dir, args.output_file)
        eval_path = os.path.join(args.myrnn_dir, 'test_word.py')
        capitalize = '--capitalize' if args.capitalize else ''
        cmd = 'python {} --data {} --model {} --output {} --ignore-eos --gpu {} {}'.format(
            eval_path, test_path, args.model, lm_output_path, args.gpu, capitalize)
        print(cmd)
        os.system(cmd)
        results = score_rnn(lm_output_path)
    else:
        logging.info("Testing RNN...")
        os.system('../example_scripts/test.sh '+ args.template_dir + ' ' +  args.model + ' ' + args.lm_data + ' ' + args.output_file + ' > '+ lm_output_path)
        results = score_rnn(lm_output_path)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

def score_unigram():
    logging.info("Scoring unigram...")
    fin = open("unigram.output", 'r')
    all_scores = {}
    sent = ""
    prevLineEmpty = True
    i = 0
    for line in fin:
        if "p( " in line:
            word = line.split("p( ")[1].split(" |")[0]
            score = float(line.split("[ ")[-1].split(" ]")[0])
            if word not in all_scores:
                all_scores[word] = score
    fin.close()
    return all_scores

def score_ngram():
    fin = open("ngram.output", 'r')
    all_scores = {}
    i = 0
    finished = True
    sent = []
    prev_sentid = -1
    for line in fin:
        if "p(" in line:
            finished = False
        if not finished and "</s>" not in line:
            word = line.split("p( ")[1].split(" |")[0]
            score = float(line.split("[ ")[-1].split(" ]")[0])
            sent.append((word,score))
            if word == "<eos>":
                name_found = False
                for (k1,v1) in sorted(name_lengths.items(), key=operator.itemgetter(1)):
                    if i < v1 and not name_found:
                        name_found = True
                        if k1 not in all_scores:
                            all_scores[k1] = {}
                        key_found = False
                        for (k2,v2) in sorted(key_lengths[k1].items(), key=operator.itemgetter(1)):
                            if i <  v2 and not key_found:
                                key_found = True
                                if k2 not in all_scores[k1]:
                                    all_scores[k1][k2] = []
                                all_scores[k1][k2].append(sent)
                sent = []
                if i != prev_sentid+1:
                    logging.info("Error at sents "+sentid+" and "+prev_sentid)
                prev_sentid = i
                finished = True
                i += 1
        else:
            finished = True
    fin.close()
    return all_scores


def score_rnn(score_fn):
    logging.info("Scoring RNN...")
    with open(score_fn, 'r') as f:
        all_scores = {}
        first = False
        score = 0.
        sent = []
        prev_sentid = -1
        for line in f:
            if line.strip() == "":
                first = True
            elif "===========================" in line:
                first = False
                break
            elif first and len(line.strip().split()) == 6 and "torch.cuda" not in line:
                wrd, sentid, wrd_score = [line.strip().split()[i] for i in [0,1,4]]
                score = -1 * float(wrd_score) # multiply by -1 to turn surps back into logprobs
                sent.append((wrd, score))
                if wrd == ".":
                    name_found = False
                    for (k1,v1) in sorted(name_lengths.items(), key=operator.itemgetter(1)):
                        if float(sentid) < v1 and not name_found:
                            name_found = True
                            if k1 not in all_scores:
                                all_scores[k1] = {}
                            key_found = False
                            for (k2,v2) in sorted(key_lengths[k1].items(), key=operator.itemgetter(1)):
                                if int(sentid) <  v2 and not key_found:
                                    key_found = True
                                    if k2 not in all_scores[k1]:
                                        all_scores[k1][k2] = []
                                    all_scores[k1][k2].append(sent)
                    sent = []
                    if float(sentid) != prev_sentid+1:
                        logging.info("Error at sents "+sentid+" and "+prev_sentid)
                    prev_sentid = float(sentid)
    return all_scores

def clean_files(mode):
    if args.model_type.lower() == 'ngram':
        os.system('rm ngram.output unigram.output')
    else:
        os.system('rm rnn.output')


test_LM()
