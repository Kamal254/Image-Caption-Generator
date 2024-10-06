import os
import regex as re
from sourcecode.model.utils.utils import create_dir
from sourcecode import logger
import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from pickle import dump, load
import pickle

class PreprocessingData():
    def __init__(self):
        pass

    def preprocess_text_data(self) ->dict:
        flickr_dataset = "../../../artifacts/data"
        with open(os.path.join(flickr_dataset, 'captions.txt'), 'r') as f:
            next(f)
            caption_file = f.read()
        logger.info("Reading Caption file")
        mapping = {}
        for line in caption_file.split('\n'):
            tokens = re.split(r'\.jpg,', line)
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], tokens[1]
            image_id = image_id.strip()
            caption = caption.strip()
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)

        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                # lowercase the caption
                caption = caption.lower()
                caption = caption.replace('[^A-Za-z]', '')
                caption = caption.replace('\s+', ' ')
                caption = 'startseq ' + " ".join([words for words in caption.split() if len(words)>1])+ ' endseq'
                captions[i] = caption

        caption_list = []
        for key in mapping:
            for caption in mapping[key]:
                caption_list.append(caption)

        logger.info(f"length of the caption list is {len(caption_list)}")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(caption_list)
        vocab_size = len(tokenizer.word_index) + 1
        logger.info(f'vocab size {vocab_size}')
        max_length = max(len(caption.split()) for caption in caption_list)
        logger.info(f'Found Max length of the caption {max_length}')
        saved_data = "../../../artifacts/saved_data"
        create_dir([saved_data])
        with open('../../../artifacts/saved_data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        with open('../../../artifacts/saved_data/mapping.pkl', 'wb') as f:
            pickle.dump(mapping, f)
    
preprocessdata = PreprocessingData()
preprocessdata.preprocess_text_data()