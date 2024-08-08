import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataset_preparation.AIC20.step01_prep_videos_annotations import main_prep_videos_annotations
from dataset_preparation.AIC20.step02_extract_frames import main_extract_frames
from dataset_preparation.AIC20.step03_preprocess_annotations import main_preprocess_annotations
from dataset_preparation.AIC20.step04_filter_sc_tracking import main_filter_sc_tracking
from dataset_preparation.AIC20.step05_extract_reid_embeddings import main_extract_reid_embeddings
from dataset_preparation.AIC20.step05b_extract_galleries import main_extract_galleries
from tuning.step06_hyperparameter_tuning import main_training_hp_tuning

def run_full_chain(args:argparse.Namespace):
    """Runs the selected steps from the full chain: preprocessing + training

    Args:
        args (argparse.Namespace): Contains the tasks selected to be run and the path to the configuration file
    """
    config_file = args.config_file
    all_steps = args.all
    preprocessing = args.preprocessing
    training = args.training
    s1 = args.s1
    s3 = args.s3
    s4 = args.s4
    s5 = args.s5
    s5b = args.s5b

    if all_steps or preprocessing or s1:
        # step 01:
        main_prep_videos_annotations(config_file)
        # step 02
        main_extract_frames(config_file)
    
    if all_steps or preprocessing or s3:
        # step 03
        main_preprocess_annotations(config_file)
    
    if all_steps or preprocessing or s4:
        #step 04  
        main_filter_sc_tracking(config_file)

    if all_steps or preprocessing or s5:
        # step 05
        main_extract_reid_embeddings(config_file)
    
    if all_steps or preprocessing or s5b:
        # step 05b - galleries
        main_extract_galleries(config_file)
    
    if all_steps or training:
        # step 06 HP tuning and training
        main_training_hp_tuning(config_file) 
        
     
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MCMT GNN',
                    description='Multi-Camera Multi-Object Tracking using Graph Neural Networks.',
                    epilog='2024',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
                    )
    parser.add_argument("-c", "--config-file", metavar="path", default=os.path.join("config", "configuration.yml"), help="Path to yml configuration file")
    group = parser.add_argument_group("Task", "Select the task to run. Only one task from the following list can be selected.")
    exgr = group.add_mutually_exclusive_group(required=True)
    exgr.add_argument("--all", help="Runs all preprocessing steps and HP search / training", action="store_true")
    exgr.add_argument("-p", "--preprocessing", help="Runs all preprocessing steps", action="store_true")
    exgr.add_argument("-t", "--training", help="Runs HP search and training", action="store_true")
    exgr.add_argument("-s1", help="Runs preprocessing steps 1 and 2", action="store_true")
    exgr.add_argument("-s3", help="Runs preprocessing step 3", action="store_true")
    exgr.add_argument("-s4", help="Runs preprocessing step 4", action="store_true")
    exgr.add_argument("-s5", help="Runs preprocessing step 5", action="store_true")
    exgr.add_argument("-s5b", help="Runs preprocessing step 5b", action="store_true")


    args = parser.parse_args()   

    # Run full chain 
    run_full_chain(args)