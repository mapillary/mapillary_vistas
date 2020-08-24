# Copyright (c) Facebook, Inc. and its affiliates.

try:
    from tqdm import tqdm as progress
except ImportError:
    # tqdm sometimes messes up the output.
    # this function enables a quite fallback in case tqdm is not installed
    def progress(iterable, **kwargs):
        return iterable
