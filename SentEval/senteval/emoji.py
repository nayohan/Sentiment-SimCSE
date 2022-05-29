# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
emoji - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
from datasets import load_dataset

from senteval.tools.validation import SplitClassifier


class EmojiEval(object):
    def __init__(self, task_path, nclasses=20, seed=1111):
        self.seed = seed

        # binary of fine-grained
        #assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : Emoji %s classification *****\n\n', self.task_name)
    
        dataset = load_dataset("tweet_eval", "emoji", script_version="master")
        print(dataset)
        train =  {'X': [e.split() for e in dataset['train']['text']], 'y': dataset['train']['label']} 
        dev = {'X': [e.split() for e in dataset['validation']['text']], 'y': dataset['validation']['label']} 
        test = {'X': [e.split() for e in dataset['test']['text']], 'y': dataset['test']['label']} 
        # print('emoji')
        # print('train:', train['X'][0], train['y'][0])
        # print('dev:', dev['X'][0], dev['y'][0])
        # print('test:', test['X'][0], test['y'][0])

        #train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        #dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        #test = self.loadFile(os.path.join(task_path, 'sentiment-test'))
        self.emoji_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.emoji_data['train']['X'] + self.emoji_data['dev']['X'] + \
                  self.emoji_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        emoji_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    emoji_data['y'].append(int(sample[1]))
                    emoji_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    emoji_data['y'].append(int(sample[0]))
                    emoji_data['X'].append(sample[1].split())
        assert max(emoji_data['y']) == self.nclasses - 1
        return emoji_data

    def run(self, params, batcher):
        emoji_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.emoji_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.emoji_data[key]['X'],
                                     self.emoji_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.emoji_data[key]['X'], self.emoji_data[key]['y'] = map(list, zip(*sorted_data))

            emoji_embed[key]['X'] = []
            for ii in range(0, len(self.emoji_data[key]['y']), bsize):
                batch = self.emoji_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                emoji_embed[key]['X'].append(embeddings)
            emoji_embed[key]['X'] = np.vstack(emoji_embed[key]['X'])
            emoji_embed[key]['y'] = np.array(self.emoji_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        clf = SplitClassifier(X={'train': emoji_embed['train']['X'],
                                 'valid': emoji_embed['dev']['X'],
                                 'test': emoji_embed['test']['X']},
                              y={'train': emoji_embed['train']['y'],
                                 'valid': emoji_embed['dev']['y'],
                                 'test': emoji_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            EMOJI {2} classification\n'.format(devacc, testacc, self.task_name))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(emoji_embed['dev']['X']),
                'ntest': len(emoji_embed['test']['X'])}
