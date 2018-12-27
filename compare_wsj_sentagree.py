import os
import sys
import logging
from runexp import Workflow

full_model_prefix = '../multitask-lm/exp/wsj/models/sentagree'
simple_model_prefix = '../multitask-lm/exp/wsj/models/sentagree/ignore_simple/'

full_models = ['mode=sentence.pt',
               'margin=1.mode=sentagree.pt',
               'margin=3.mode=sentagree.pt',
               'margin=5.mode=sentagree.pt',
               'margin=10.mode=sentagree.pt',
               'margin=30.mode=sentagree.pt',
               'margin=50.mode=sentagree.pt',
               'margin=1.mode=sentagree.upsample-agreement.pt',
               'margin=3.mode=sentagree.upsample-agreement.pt',
               'margin=5.mode=sentagree.upsample-agreement.pt',
               'margin=10.mode=sentagree.upsample-agreement.pt',
               'margin=30.mode=sentagree.upsample-agreement.pt',
               'margin=50.mode=sentagree.upsample-agreement.pt']

simple_models = ['margin=1.mode=sentagree.pt',
                 'margin=3.mode=sentagree.upsample-agreement.pt',
                 'margin=5.mode=sentagree.upsample-agreement.pt',
                 'margin=30.mode=sentagree.pt',
                 'margin=3.mode=sentagree.pt',
                 'margin=5.mode=sentagree.pt',
                 'margin=10.mode=sentagree.pt',
                 'margin=50.mode=sentagree.pt']

full_model_paths = [os.path.join(full_model_prefix, m) for m in full_models]
simple_model_paths = [os.path.join(simple_model_prefix, m) for m in simple_models]

job_i = 0
def mk_tasks(exp):
    def helper(is_full = True):
        global job_i
        attr = 'full' if is_full else 'simple'
        prefix = full_model_prefix if is_full else simple_model_prefix
        models = full_models if is_full else simple_models

        for model in models:
            src = os.path.join(prefix, model)
            lm_output = 'lm_output/sentagree/wsj/{}/{}'.format(attr, model)
            tgt = '{}/results.pickle'.format(lm_output)

            wait = job_i * 30
            job_i += 1
            exp(name = tgt,
                source = src,
                target = tgt,
                rule = 'sleep {}; python src/LM_eval.py --model {} '
                '--model_type myrnn --template_dir default-templates '
                '--myrnn_dir ../multitask-lm/lm --lm_output {}'.format(
                    wait, src, lm_output))

            analyze_output = 'results/sentagree/wsj/{}/{}'.format(attr, model)
            analyze_tgt = '{}/rnn/full_sent/overall_accs.txt'.format(analyze_output)
            exp(name = analyze_tgt,
                source = tgt,
                target = analyze_tgt,
                rule = 'python src/analyze_results.py --results_file {} '
                '--model_type rnn --out_dir {} --mode full'.format(
                    tgt, analyze_output))

    helper(True)
    helper(False)

logger = logging.getLogger(__name__)
logger.debug('set tasksk')

exp = Workflow()

mk_tasks(exp)

logger.debug('run tasks')
if not exp.run():
    sys.exit(1)

