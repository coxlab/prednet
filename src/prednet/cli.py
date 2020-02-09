"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mprednet` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``prednet.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``prednet.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import argparse
import os.path
import pkg_resources

import jupyterlab.labapp

parser = argparse.ArgumentParser(description='Train and evaluate on single video.')
# parser.add_argument('train_or_predict', nargs='?', choices=('train', 'predict', 'jupyter'), help="Only train or only evaluate.")
# There appears to be no native way in argparse to have an optional positional argument prior to a list of positional arguments.
parser.add_argument('--model-file', help="Path to load and/or save the model file.")
parser.add_argument('--subsequence-length', type=int,
                    help="Length of subsequences (at the end of each subsequence, PredNet forgets its state and starts fresh).")
parser.add_argument('--number-of-epochs', type=int, default=150, help="Number of epochs to use in training.")
parser.add_argument('--steps-per-epoch', type=int, default=125, help="Steps per epoch to use in training.")
subparsers = parser.add_subparsers(help='We can only train, or predict, or simply open a Jupyter instance.', dest='subparser_name',
                                   #required=True,
                                  )
# We'd like to allow users to skip specifying the subparser (default to predict) and just provide video files,
# but that gets hairy since the parser will first try to interpret the video file path as a subparser name.
# This is more explicit anyway.
# But requiring subparsers requires Python 3.7, so that has to wait until we upgrade to TensorFlow 2.
trainParser = subparsers.add_parser('train', help='Train a model.')
predictParser = subparsers.add_parser('predict', help='Predict frames using a model (train if necessary).')
jupyterParser = subparsers.add_parser('jupyter', help='Launch a Jupyter instance showing how you can use PredNet programmatically.')
trainParser.add_argument('paths_to_videos', nargs='+', help="Paths to source video files.")
predictParser.add_argument('paths_to_videos', nargs='+', help="Paths to source video files.")
jupyterParser.add_argument('jupyter_lab_arguments', nargs='*',
                           help="All further arguments will be passed through to jupyter lab just as if you invoked jupyter lab directly.")
# jupyterParser.add_argument('--port', type=int, default=8888, help="The port the notebook server will listen on.")
# jupyterParser.add_argument('--ip', default='localhost', help="The IP address the notebook server will listen on. If you get OSError: [Errno 99] Cannot assign requested address, try --ip 127.0.0.1. That shouldn't matter but sometimes does with odd proxy/firewall configurations.")
# IPython listens on localhost by default. 127.0.0.1 ought to behave the same, and does in almost all cases. Some cases where this can be handled differently include local proxies and/or firewalls (usually due to a configuration oversight, rather than an intentional difference in behavior). We have found cases where localhost works and 127 doesn't and vice versa, so there isn't an unambiguously correct answer for the default.
# https://github.com/ipython/ipython/issues/6193


def main(args=None):
    args = parser.parse_args(args=args)
    if args.subparser_name == 'jupyter': assert 'paths_to_videos' not in vars(args)
    if 'paths_to_videos' in vars(args) and len(args.paths_to_videos) != 1 and not args.model_file:
      parser.error('If you want to specify more than one video, we cannot infer the name you want for the model file.')
    if args.subparser_name == 'jupyter':
      # open an included notebook showing how to do something simple with prednet
      jupyterlab.labapp.main([pkg_resources.resource_filename(__name__, os.path.join('resources', 'quickstart.ipynb')),
                              *args.jupyter_lab_arguments])
    else:
      # We put off importing prednet because there's a delay when the TensorFlow backend is loaded.
      # The TensorFlow backend will be loaded as soon as we import either, so we might as well import both.
      import prednet.train
      import prednet.evaluate
      if args.subparser_name == 'train':
          prednet.train.train_on_video_list(args.paths_to_videos,
                                            number_of_epochs=args.number_of_epochs,
                                            steps_per_epoch=args.steps_per_epoch)
      elif args.subparser_name == 'predict':
        # This might not be the best way to do things, but this function will
        # automatically use the saved model if the file(s) already exist.
        prednet.evaluate.save_predicted_frames_for_video_list(
            args.paths_to_videos,
            number_of_epochs=args.number_of_epochs, steps_per_epoch=args.steps_per_epoch,
            model_file_path=args.model_file,
            nt=args.subsequence_length,
            )

