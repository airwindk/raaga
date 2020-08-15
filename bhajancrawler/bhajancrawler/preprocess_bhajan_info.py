import json
import os

os.chdir('..')
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce
from pathlib import Path
import hashlib
import src.config as config
import json


class RaagaPreprocessor:

    @staticmethod
    def clean(raaga_list, **kwargs):
        options = {'remove': ['Major (', 'Minor (', '~', ')']}
        options.update(kwargs)

        raaga_list_clean = [
            reduce(lambda raw, old: raw.replace(old, ''), [raaga] + options['remove'])
            for raaga in raaga_list
        ]
        raaga_list_clean = [raaga.strip() for raaga in raaga_list_clean]
        return raaga_list_clean

    @staticmethod
    def filter(raaga_list, **kwargs):
        options = {'min_count': config.RAAGA_COUNT_THRESH}
        options.update(kwargs)

        final_raagas = {raaga: count for raaga, count in Counter(raaga_list).items() if count > options['min_count']}
        raaga_list_filt = [raaga for raaga in raaga_list if raaga in final_raagas.keys()]
        return raaga_list_filt

    @staticmethod
    def raaga_to_idx(raaga_list, **kwargs):
        options = {'save_to_file': True,
                   'dir': 'data/raw/'}
        options.update(kwargs)

        raagas = list(set(raaga_list))
        raaga_map = {raaga: idx for idx, raaga in enumerate(raagas)}

        if options['save_to_file']:
            with open(os.path.join(options['dir'],"raaga_map.json"), "w+") as f:
                json.dump(raaga_map, f)

        return raaga_map


class Formatter:

    def __init__(self, file=config.CRAWLER_SETTINGS['file_uri']):
        self.file = file


    @staticmethod
    def _get_hash(str_value):
        hash_object = hashlib.sha1(str_value.encode('utf-8'))
        hex_dig = hash_object.hexdigest()
        return hex_dig

    @staticmethod
    def _get_audio_file_name(url, dir="data/raw/bhajans_audio/"):
        file_format = Path(url).suffix
        file_name = Formatter._get_hash(url) + file_format
        return os.path.join(dir, file_name)

    def format_bhajan(self, **kwargs):
        options = {'input_dir': self.file,
                   'save_to_file': True,
                   'output_dir': "data/processed/bhajans_info_cleaned.json"
                   }
        options.update(kwargs)

        with open(options['input_dir'], "r+") as f:
            bhajans = json.loads(f.read())

            # clean and filter raagas
            raaga_list = RaagaPreprocessor.clean([bhajan['raaga'] for bhajan in bhajans])
            final_raagas = set(RaagaPreprocessor.filter(raaga_list))
            bhajans = [bhajan.update({'raaga': raaga})
                       for raaga, bhajan in zip(raaga_list, self.bhajans) if raaga in final_raagas
                       ]

            # generate audio file name and add to dict
            audio_file_names = [Formatter._get_audio_file_name(bhajan['link']) for bhajan in bhajans]
            self.bhajans = [bhajan.update({'file_name': file}) for file, bhajan in zip(audio_file_names, bhajans)]


        if options['save_to_file']:
            with open(options['output_dir'], "w+") as f:
                json.dump(self.bhajans, f)

