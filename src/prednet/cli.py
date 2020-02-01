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
import jupyterlab.labapp
import prednet.train
import prednet.evaluate

parser = argparse.ArgumentParser(description='Train and evaluate on single video.')
parser.add_argument('paths_to_videos', nargs='+', help="Path to source video file.")
parser.add_argument('--model-file', help="Path to load and/or save the model file.")
parser.add_argument('--subsequence-length', type=int,
                    help="Length of subsequences (at the end of each subsequence, PredNet forgets its state and starts fresh).")
parser.add_argument('--number-of-epochs', type=int, default=150, help="Number of epochs to use in training.")
parser.add_argument('--steps-per-epoch', type=int, default=125, help="Steps per epoch to use in training.")
parser.add_argument('train_or_predict', nargs='?', choices=('train', 'predict', 'jupyter'), help="Only train or only evaluate.")


def main(args=None):
    args = parser.parse_args(args=args)
    if len(args.paths_to_videos) != 1 and not args.model_file:
      parser.error('If you want to specify more than one video, we cannot infer the name you want for the model file.')
    if args.train_or_predict:
      if args.train_or_predict == 'train':
          prednet.train.train_on_video_list(args.paths_to_videos,
                                            number_of_epochs=args.number_of_epochs,
                                            steps_per_epoch=args.steps_per_epoch)
      elif args.train_or_predict == 'predict':
        # This might not be the best way to do things, but this function will
        # automatically use the saved model if the file(s) already exist.
        prednet.evaluate.save_predicted_frames_for_video_list(
            args.paths_to_videos,
            number_of_epochs=args.number_of_epochs, steps_per_epoch=args.steps_per_epoch,
            model_file_path=args.model_file,
            nt=args.subsequence_length,
            )
      elif args.train_or_predict == 'jupyter':
        jupyterlab.labapp.main()  # open an included notebook showing how to do something simple with prednet
    else:
        prednet.evaluate.save_predicted_frames_for_video_list(
            args.paths_to_videos,
            number_of_epochs=args.number_of_epochs, steps_per_epoch=args.steps_per_epoch,
            model_file_path=args.model_file,
            nt=args.subsequence_length,
            )
