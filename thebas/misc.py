# coding=utf-8
"""A jumble of seemingly useful stuff."""
import os.path as op
import os


def home():
    #What is the equivalent of user.home in python 3?
    return op.expanduser('~')


def ensure_writable_dir(path):
    """Ensures that a path is a writable directory."""
    def check_path(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    if op.exists(path):
        check_path(path)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if op.exists(path):  # Simpler than using a file lock to work on multithreading...
                check_path(path)
            else:
                raise


def ensure_dir(path):
    ensure_writable_dir(path)