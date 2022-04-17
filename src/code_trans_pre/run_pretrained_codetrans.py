import sys
import os
import torch
import time
import numpy as np

from src.utils import utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


"""
    https://github.com/agemagician/CodeTrans
"""

if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    corpus_name = 'huetal'
    # corpus_name = 'codexglue'

    preproc_config_name = 'none_none'
    # preproc_config_name = 'none_camelsnakecase'

    test_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config_name}.csv'

    size_threshold = 20

    max_code_len = 200
    max_len_desc = 30

    num_beams = 10

    generated_desc_dir = f'../../resources/descriptions/'

    generated_desc_dir = os.path.join(generated_desc_dir, lang, corpus_name)

    # model_name = 'codetrans_small'
    # model_name = 'codetrans_base'

    model_name = 'codetrans_mt_tf_base'
    # model_name = 'codetrans_mt_tf_small'
    # model_name = 'codetrans_mt_tf_large'

    # model_name = 'codetrans_mt_base'
    # model_name = 'codetrans_mt_small'
    # model_name = 'codetrans_mt_large'

    # model_name = 'codetrans_tf_base'
    # model_name = 'codetrans_tf_small'
    # model_name = 'codetrans_tf_large'

    model_path = None

    if lang == 'python':

        if model_name == 'codetrans_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python'
        elif model_name == 'codetrans_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python'
        elif model_name == 'codetrans_mt_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'codetrans_mt_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_multitask'
        elif model_name == 'codetrans_mt_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python_multitask'
        elif model_name == 'codetrans_mt_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask'
        elif model_name == 'codetrans_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_transfer_learning_finetune'
        elif model_name == 'codetrans_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python_transfer_learning_finetune'
        elif model_name == 'codetrans_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_python_transfer_learning_finetune'

    elif lang == 'java':

        if model_name == 'codetrans_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java'
        elif model_name == 'codetrans_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java'
        elif model_name == 'codetrans_mt_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'codetrans_mt_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask'
        elif model_name == 'codetrans_mt_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java_multitask'
        elif model_name == 'codetrans_mt_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_java_multitask'
        elif model_name == 'codetrans_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_transfer_learning_finetune'
        elif model_name == 'codetrans_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java_transfer_learning_finetune'
        elif model_name == 'codetrans_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_java_transfer_learning_finetune'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nDevice:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path,
                                            sample_size=size_threshold)
    test_codes = test_data[0]

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0])

    print('\nModel:', model_name, '\n')

    tokenizer = AutoTokenizer.from_pretrained(model_path, skip_special_tokens=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    total_examples = len(test_codes)

    generated_descriptions = []

    execution_times = []

    with tqdm(total=total_examples, file=sys.stdout,
              desc='  Generating summaries') as pbar:

        for i in range(total_examples):

            code = test_codes[i]

            start = time.time()

            code_seq = tokenizer.encode(code,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=max_code_len).to(device)

            desc_ids = model.generate(code_seq,
                                      max_length=max_len_desc,
                                      num_beams=num_beams,
                                      early_stopping=True)

            description = \
                [tokenizer.decode(g, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
                 for g in desc_ids]

            description = description[0].strip()

            end = time.time()

            execution_time = end - start

            execution_times.append(execution_time)

            generated_descriptions.append(description)

            pbar.update(1)

    generated_desc_file = os.path.join(generated_desc_dir, f'pre_{model_name}_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))

    execution_stats = f'time: {np.mean(execution_times)} -- std: {np.std(execution_times)}'

    execution_stats_file = os.path.join(generated_desc_dir,
                                        f'pre_{model_name}_{corpus_name}_{preproc_config_name}_time.txt')

    with open(execution_stats_file, 'w') as file:
        file.write(execution_stats)
