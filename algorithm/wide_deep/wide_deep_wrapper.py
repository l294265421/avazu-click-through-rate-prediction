from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import os

import util.common_util as cu
from preprocess.gen_vocabulary import gen_vocabulary_file_path

import tensorflow as tf
import pandas as pd
import tensorflow.contrib.slim as slim

import math
from data.raw_data import base_dir

class WideDeepClassifier:
    def __init__(self, category, all_columns, column_defaults, numerical_features, categorical_features, train_filename, test_filename, cross=[], mode_type='wide', train_epochs=2, batch_size=8):
        self.category = category
        self.all_columns = all_columns
        self.column_defaults = column_defaults
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.cross = cross
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_dir', type=str, default='model/wide_deep',
            help='Base directory for the model.')
        parser.add_argument(
            '--model_type', type=str, default=mode_type,
            help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
        parser.add_argument(
            '--train_epochs', type=int, default=train_epochs, help='Number of training epochs.')
        parser.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='The number of training epochs to run between evaluations.')
        parser.add_argument(
            '--batch_size', type=int, default=batch_size, help='Number of examples per batch.')
        parser.add_argument(
            '--train_data', type=str, default=base_dir + train_filename,
            help='Path to the training data.')

        parser.add_argument(
            '--test_data', type=str, default=base_dir + test_filename,
            help='Path to the test data.')

        tf.logging.set_verbosity(tf.logging.INFO)
        FLAGS, unparsed = parser.parse_known_args()
        self.FLAGS = FLAGS

    def categorical_column_with_vocabulary_list(self, row_column_name):
        return tf.feature_column.categorical_column_with_vocabulary_file(row_column_name, gen_vocabulary_file_path(base_dir, self.category, row_column_name))

    def build_model_columns(self):
        categorical_columns = []

        for column in self.categorical_features:
            categorical_columns.append(self.categorical_column_with_vocabulary_list(column))

        numerical_columns = []
        for column in self.numerical_features:
            numerical_columns.append(tf.feature_column.numeric_column(column, normalizer_fn=slim.batch_norm))

        crossed_columns = []
        for cross_feature in self.cross:
            crossed_columns.append(tf.feature_column.crossed_column(keys=cross_feature[:-1], hash_bucket_size=cross_feature[-1]))

        wide_columns = categorical_columns + numerical_columns + crossed_columns

        deep_categorical_columns = []
        for feature_column in categorical_columns:
            if feature_column.vocabulary_size > 50:
                deep_categorical_columns.append(tf.feature_column.embedding_column(feature_column, round(feature_column.vocabulary_size ** 0.25)))
            else:
                deep_categorical_columns.append(tf.feature_column.indicator_column(feature_column))

        deep_columns = deep_categorical_columns + numerical_columns

        return wide_columns, deep_columns

    def build_estimator(self, model_dir, model_type):
        """Build an estimator appropriate for the given model type."""
        wide_columns, deep_columns = self.build_model_columns()
        hidden_units = [128, 64, 32]

        # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
        # trains faster than GPU for this model.
        run_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'GPU': 0}))
        if model_type == 'wide':
            return tf.estimator.LinearClassifier(
                model_dir=model_dir,
                feature_columns=wide_columns,
                config=run_config)
        elif model_type == 'deep':
            return tf.estimator.DNNClassifier(
                model_dir=model_dir,
                feature_columns=deep_columns,
                hidden_units=hidden_units,
                config=run_config)
        else:
            return tf.estimator.DNNLinearCombinedClassifier(
                model_dir=model_dir,
                linear_feature_columns=wide_columns,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=hidden_units,
                config=run_config)

    def fit(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        shutil.rmtree(self.FLAGS.model_dir, ignore_errors=True)

        model = self.build_estimator(self.FLAGS.model_dir, self.FLAGS.model_type)

        model.train(input_fn=lambda: self.input_fn(self.FLAGS.train_data, self.FLAGS.train_epochs, False, self.FLAGS.batch_size, False))
        self.model = model


    def predict(self):
        if not hasattr(self, 'model'):
            self.model = self.build_estimator(self.FLAGS.model_dir, self.FLAGS.model_type)
        predictions = list(
            self.model.predict(input_fn=lambda: self.input_fn(self.FLAGS.test_data, 1, False, self.FLAGS.batch_size, True)))
        predicted_logistics = [float('%.7f' % (p["logistic"][0])) for p in predictions]
        # predicted_logistics = [float('%.7f' % (p["predictions"][0])) for p in predictions]
        submission_id = pd.read_csv(self.FLAGS.test_data, usecols=['id'], dtype={'id': str})['id']
        result = pd.DataFrame({'id': submission_id, 'click': predicted_logistics})[['id', 'click']]
        result.to_csv(base_dir + self.category + '_submit.csv', index=False)

    def input_fn(self, data_file, num_epochs, shuffle, batch_size, predict):
        """Generate an input function for the Estimator."""
        assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have either run data_download.py or '
            'set both arguments --train_data and --test_data.' % data_file)

        def parse_csv(value):
            print('Parsing', data_file)
            columns = tf.decode_csv(value, record_defaults=self.column_defaults)
            features = dict(zip(self.all_columns, columns))

            features.pop('id')
            if not predict:
                return features, features.pop('click')
            else:
                return features

        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(data_file)
        # Skip header row
        dataset = dataset.skip(1)

        # if shuffle:
            # dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

        dataset = dataset.map(parse_csv, num_parallel_calls=5)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()