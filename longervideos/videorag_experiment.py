import os
import json
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

import argparse
parser = argparse.ArgumentParser(description="Set sub-category and CUDA device.")
parser.add_argument('--collection', type=str, default='4-rag-lecture')
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
sub_category = args.sub_category

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import *
from videorag import VideoRAG, QueryParam

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    ## learn
    video_base_path = f'./{sub_category}/videos/'
    video_files = sorted(os.listdir(video_base_path))
    video_paths = [os.path.join(video_base_path, f) for f in video_files]
    videorag = VideoRAG(cheap_model_func=gpt_4o_mini_complete, best_model_func=gpt_4o_mini_complete, working_dir=f"./videorag-workdir/{sub_category}")
    videorag.insert_video(video_path_list=video_paths)
    
    ## inference
    with open(f'./dataset.json', 'r') as f:
        longervideos = json.load(f)
    
    videorag = VideoRAG(cheap_model_func=gpt_4o_mini_complete, best_model_func=gpt_4o_mini_complete, working_dir=f"./videorag-workdir/{sub_category}")
    videorag.load_caption_model(debug=False)
    
    answer_folder = f'./videorag-answers/{sub_category}'
    os.makedirs(answer_folder, exist_ok=True)
    
    collection_id = sub_category.split('-')[0]
    querys = longervideos[collection_id][0]['questions']
    for i in range(len(querys)):
        query_id = querys[i]['id']
        query = querys[i]['question']
        param = QueryParam(mode="videorag")
        param.wo_reference = True
        print("Query: ", query)
        
        response = videorag.query(query=query, param=param)
        print(response)
        with open(os.path.join(answer_folder, f'/answer_{query_id}.md'), 'w') as f:
            f.write(response)