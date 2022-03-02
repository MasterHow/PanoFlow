from opticalflow.core.dataset import KITTIDemoManager


def preprocess_data(np_raw_data, args):
    return KITTIDemoManager.preprocess_data(np_raw_data, args)
