import argparse
import json

def get_args(all_models):
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  ''' What To Do '''
  parser.add_argument('--test', dest='for_what', 
    action='store_const', const='test', help='Test')
  parser.add_argument('--target-modules', dest='target_modules', 
    default=all_models, nargs="*", choices=all_models, 
    help='Modules to train')

  ''' Input Directories and Files '''
  parser.add_argument('--dataset', dest='dataset', required=True,
    choices=['mars','duke'],
    help='Dataset to be used')
  parser.add_argument('--json-path', dest='json_path', required=True,
    help='Path to dataset JSON file')
  parser.add_argument('--h5-dir', dest='h5_dir', required=True,
    help='Directory that contains dataset H5 files')
  parser.add_argument('--checkpoint-files', dest='checkpoint_files', 
    default='{ }', type=json.loads, 
    help='JSON string to indicate checkpoint file for each module. '
         'Beside module names like encoder, and viewclassifier, you can use '
         'special name "else", and "all".'
         'Example: {"encoder": "encodercheck.tar", "else": "allcheck.tar"}')

  ''' Model Parameters '''
  parser.add_argument('--classifier-out-size', dest='classifier_out_size',
    type=int, default=2, choices=[1, 2],
    help='Whether to use binary_cross_entropy or crossentropy')

  ''' Dataloaders '''
  parser.add_argument('--max-sample-length', dest='max_sample_length', 
    default=4, type=int,
    help='Maximum length of sample')
  parser.add_argument('--samples-per-batch', dest='samples_per_batch', 
    default=1, type=int,
    help='Number of samples per batch. Used to distribute positive and negative'
         ' samples when training. Used only for training. Batch size will be '
         'automatically divided with this value to match the total #batches.')
  parser.add_argument('--visual-transform', dest='visual_transform',
    default='normalize', choices=[None, 'normalize'],
    help='Transform to apply in IDwithPose dataset')
  parser.add_argument('--num-workers', dest='num_workers', default=1, type=int,
    help='Number of workers to load train/test data samples')
  parser.add_argument('--batch-size', dest='batch_size',
    type=int, default=32, help='Input minibatch size')

  args = parser.parse_args()
  
  params = str(vars(args))

  # Sanitize to print arguments
  s = params.find('target_modules\': [') # handle a list
  e = params.find(']', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  s = params.find('checkpoint_files\': {') # handle a dict
  e = params.find('}', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  params = sorted( params[1:-1].replace("'","").split(', ') )
  print( "\nRunning with following parameters: \n  {}\n".format('\n  '.join(params)) )

  return args