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
import prednet.evaluate

parser = argparse.ArgumentParser(description='Train and evaluate on single video.')
parser.add_argument('path_to_video', help="Path to video file.")
parser.add_argument('--model-file', help="Path to load and/or save the model file.")
parser.add_argument('--number-of-epochs', type=int, default=150, help="Number of epochs to use in training.")
parser.add_argument('--steps-per-epoch', type=int, default=125, help="Steps per epoch to use in training.")


def main(args=None):
    args = parser.parse_args(args=args)
    prednet.evaluate.save_predicted_frames_for_single_video(
        args.path_to_video,
        number_of_epochs=args.number_of_epochs, steps_per_epoch=args.steps_per_epoch)
