import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import spacy

code_txt = open('debug/total/code.original_subtoken', encoding='utf-8').read().split('\n')
summ_txt = open('debug/total/javadoc.original', encoding='utf-8').read().split('\n')

raw_data = {'Code': [line for line in code_txt[1:1000]],
            'Text': [line for line in summ_txt[1:1000]]}

df = pd.DataFrame(raw_data, columns=['Code', 'Text'])

train, test = train_test_split(df, test_size=0.2)

train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)

## tokenize
tokenize = lambda x : x.split()

"""
spacy_en = spacy.load('en')
def tokenize(text):
    a = [tok.text for tok in spacy_en.tokenizer(text)]
    return a
"""
code = Field(sequential=True, 
            use_vocab=True, 
            tokenize=tokenize, 
            lower=True,
            init_token='<sos>',
            eos_token='<eos>',
            pad_token='<pad>')

text = Field(sequential=True, 
            use_vocab=True, 
            tokenize=tokenize, 
            lower=True,
            init_token='<sos>',
            eos_token='<eos>',
            pad_token='<pad>')

fields = {'Code' : ('code', code), 'Text' : ('text', text)}

train_data, test_data = TabularDataset.splits(path='',
                                            train='train.json',
                                            test='test.json',
                                            format='json',
                                            fields=fields)

print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())

code.build_vocab(train_data, max_size=10000, min_freq=1)
text.build_vocab(train_data, max_size=10000, min_freq=1)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=256,
    device = "cuda"
)

for batch in train_iterator:
    print(batch)