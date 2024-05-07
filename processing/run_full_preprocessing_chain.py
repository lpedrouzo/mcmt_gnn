import argparse
import os
from dataset_preparation.AIC20.step01_prep_videos_annotations import main_prep_videos_annotations
from dataset_preparation.AIC20.step02_extract_frames import main_extract_frames
from dataset_preparation.AIC20.step03_preprocess_annotations import main_preprocess_annotations
from dataset_preparation.AIC20.step04_filter_sc_tracking import main_filter_sc_tracking
from dataset_preparation.AIC20.step05_extract_reid_embeddings import main_extract_reid_embeddings
from dataset_preparation.AIC20.step05b_extract_galleries import main_extract_galleries

def run_full_chain(config01:str, config02:str, config03:str, run_filters:bool, config04GT:str, config04SCT:str, config05GT:str, config05SCT:str, config05galleries:str):
    # step 01:
    main_prep_videos_annotations(config01)
    # step 02
    main_extract_frames(config02)
    # step 03
    main_preprocess_annotations(config03)
    
    if run_filters:
        print("Running filters on GT and SCT annotation data")
        # step 04 GT
        main_filter_sc_tracking(config04GT)
        # step 04 SCT
        main_filter_sc_tracking(config04SCT)
    else:
        print("No filtering is done to annotations!")
    
    # step 05 
    print("Extracting ReID embeddings for ground truth files")
    main_extract_reid_embeddings(config05GT)
    print("Extracting ReID embeddings for single-camera tracking files")
    main_extract_reid_embeddings(config05SCT)

    # step 05b - galleries
    print("Extracting galleries")
    main_extract_galleries(config05galleries)
        
     
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MCMT GNN',
                    description='Multi-Camera Multi-Object Tracking using Graph Neural Networks. Runs all preprocessing steps prior to training.',
                    epilog='2024',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
                    )
    parser.add_argument("-s1", "--step1-config", metavar="01", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 1")
    parser.add_argument("-s2", "--step2-config", metavar="02", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 2")
    parser.add_argument("-s3", "--step3-config", metavar="03", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 3")
    parser.add_argument("--no-filter", action="store_false", help="Add this flag to not run step 04 related to bounding box filtering")
    parser.add_argument("-s4GT", "--step4GT-config", metavar="04GT", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 4 for GT files")
    parser.add_argument("-s4SCT", "--step4SCT-config", metavar="04SCT", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 4 for SCT files")
    parser.add_argument("-s5GT", "--step5GT-config", metavar="05GT", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 5 for GT files")
    parser.add_argument("-s5SCT", "--step5SCT-config", metavar="05SCT", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 5 for SCT files")
    parser.add_argument("-s5g", "--step5galleries-config", metavar="05gall", default=os.path.join("..", "config", "preprocessing.yml"), help="Path to yml configuration file for step 5b (galleries)")



    args = parser.parse_args()

    config01 = args.step1_config
    config02 = args.step2_config
    config03 = args.step3_config
    run_filters = args.no_filter
    config04GT = args.step4GT_config
    config04SCT = args.step4SCT_config
    config05GT = args.step5GT_config
    config05SCT = args.step5SCT_config
    config05galleries = args.step5galleries_config
    
    
    # Run full chain 
    run_full_chain(config01, config02, config03, run_filters, config04GT, config04SCT, config05GT, config05SCT, config05galleries)