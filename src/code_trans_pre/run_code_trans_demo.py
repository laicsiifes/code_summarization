import javalang

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def tokenize_java_code(code):
    token_list = []
    tokens = list(javalang.tokenizer.tokenize(code))
    for token in tokens:
        token_list.append(token.value)
    return ' '.join(token_list)


def example_java():

    model_name = 'SEBIS/code_trans_t5_small_code_documentation_generation_java'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True)

    # code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"
    code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    print('\nCode:', code)

    tokenized_code = tokenize_java_code(code)

    print('\nTokenized code:', tokenized_code)

    code_seq = tokenizer.encode(tokenized_code, return_tensors='pt', truncation=True, max_length=256).to('cuda')

    desc_ids = model.generate(code_seq, min_length=10, max_length=30, num_beams=10, early_stopping=True)

    description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in desc_ids]

    description = description[0].strip()

    print('\nDescription:', description)

"""
    https://github.com/agemagician/CodeTrans
"""

if __name__ == '__main__':

    example_java()
