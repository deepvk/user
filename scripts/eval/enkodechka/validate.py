from encodechka_eval import tasks
from encodechka_eval.bert_embedders import embed_bert_both, get_word_vectors_with_bert
from transformers import AutoModel, AutoTokenizer

from tqdm.auto import tqdm
import argparse
import os
import numpy as np
    
INSTRUCTIONS = {
    'STS':  [
        'Это текст для поиска перефразированного текста:',
        'Это текст для поиска перефразированного текста:'
    ],
    'PI':  [
        'Это текст для поиска перефразированного текста:',
        'Это текст для поиска перефразированного текста:'
    ],
    'NLI':  [
        'Это текст для поиска перефразированного текста:',
        'Это текст для поиска перефразированного текста:'
    ],
    'SA':  ['', ''],
    'TI':  ['', ''],
    'IA':  ['', ''],
    'IC':  ['', ''],
    'ICX': ['', '']
}
    
def main(model_path, max_length, use_instructions):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.cuda()
    
    results = {
        'mean': []
    }
    
    tasks_arr = {
        'STS': tasks.STSBTask,
        'PI':  tasks.ParaphraserTask,
        'NLI': tasks.XnliTask,
        'SA':  tasks.SentimentTask,
        'TI':  tasks.ToxicityTask,
        'IA':  tasks.InappropriatenessTask,
        'IC':  tasks.IntentsTask,
        'ICX': tasks.IntentsXTask
    }
    
    for task_name in tqdm(tasks_arr):
        instr1, instr2 = '', ''
        if use_instructions:
            instr1, instr2 = INSTRUCTIONS[task_name]
        
        new_task = tasks_arr[task_name]()
        res = new_task.eval(lambda x: embed_bert_both(x, model, tokenizer, max_length), model_path, \
                           instr1=instr1, instr2=instr2)
        results[task_name] = res[0]
        results['mean'].append(res[0])
        
    results['mean'] = np.mean(results['mean'])
    print('VALIDATION COMPLETE:', results)   
    return results
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', dest='model_path', required=True, action='store', help='path to model weights')
    parser.add_argument('--max_length', dest='max_length', required=False, action='store', default=512)
    parser.add_argument('--instr', dest='instr', action='store_true', default=False)

    args = parser.parse_args()
    
    main(model_path=args.model_path, max_length=args.max_length, use_instructions=args.instr)
