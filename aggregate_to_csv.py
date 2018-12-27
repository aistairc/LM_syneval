import argparse
import os
import re
import sys

def read_results(base_dir):
    pat = re.compile(r"(margin=(\d+)\.)*(mode=([^\.]+)\.)*(upsample-agreement\.)*pt")
    def setting_name(subdir):
        m = pat.match(subdir)
        margin = m.group(2)
        mode = m.group(4)
        upsample = 'upsample' if m.group(5) else None

        keys = [mode, margin, upsample]
        keys = [k for k in keys if k]
        return '.'.join(keys)

    summaries = {} # setting name (abbrev) -> summary;
                   # each summary is also a map of key and value

    subdirs = os.listdir(base_dir)
    for subdir in subdirs:
        abbrev = setting_name(subdir)
        summary_path = os.path.join(base_dir, subdir, 'rnn/full_sent/overall_accs.txt')
        with open(summary_path) as f:
            lines = f.read().split('\n')
        def kv(line):
            sp = line.rfind(' ')
            return line[:sp-1], float(line[sp+1:])*100

        items = [kv(line) for line in lines if line]
        summaries[abbrev] = items

    return summaries

def aggregate(args):
    base_dir = args.result_dir
    summaries = read_results(base_dir)

    # Output a csv (in stdout)
    for i, (abbrev, summary) in enumerate(summaries.items()):
        if i == 0:
            print(',' + ','.join([k for k, v in summary]))
        vs = ['{:.2f}'.format(v) for k, v in summary]
        print('{},{}'.format(abbrev, ','.join(vs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str,
                        help='An example is results/sentagree/wsj/simple/')


    args = parser.parse_args()

    aggregate(args)
