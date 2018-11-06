import logging
from datetime import datetime

from scipy.io import loadmat


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def fmt_message(self, time, message, log_level):
        fmt = '[{time}  {logger_name}  {level}]  {mess}'
        fmt_message = fmt.format(time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                 logger_name=self.logger.name,
                                 level=log_level,
                                 mess=message)
        return fmt_message

    def info(self, message):
        now = datetime.now()
        log_level = 'INFO'
        self.logger.info(self.fmt_message(now, message, log_level))

    def debug(self, message):
        now = datetime.now()
        log_level = 'DEBUG'
        self.logger.debug(self.fmt_message(now, message, log_level))

    def warning(self, message):
        now = datetime.now()
        log_level = 'WARNING'
        self.logger.warning(self.fmt_message(now, message, log_level))

    def error(self, message):
        now = datetime.now()
        log_level = 'ERROR'
        self.logger.error(self.fmt_message(now, message, log_level))


def get_logger(name="tcn"):

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(name)

    return Logger(logger)


def data_generator(dataset, framework="torch"):

    logger = get_logger("utils")
    absolute_path = "/Users/admin/Documents/Diploma/tcn/mdata"
    if dataset == "JSB":
        logger.info('loading JSB data...')
        data = loadmat('{}/JSB_Chorales.mat'.format(absolute_path))
    elif dataset == "Muse":
        logger.info('loading Muse data...')
        data = loadmat('{}/MuseData.mat'.format(absolute_path))
    elif dataset == "Nott":
        logger.info('loading Nott data...')
        data = loadmat('{}/Nottingham.mat'.format(absolute_path))
    elif dataset == "Piano":
        logger.info('loading Piano data...')
        data = loadmat('{}/Piano_midi.mat'.format(absolute_path))

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    if framework == "torch":

        import torch
        from numpy import float64

        for data in [X_train, X_valid, X_test]:
            for i in range(len(data)):
                data[i] = torch.Tensor(data[i].astype(float64))

    elif framework == "tf":
        # import tensorflow as tf
        # for data in [X_train, X_valid, X_test]:
        #     for i in range(len(data)):
        #         data[i] = tf.convert_to_tensor(value=data[i], dtype=tf.float32)
        pass

    else:
        raise Exception()

    return X_train, X_valid, X_test

