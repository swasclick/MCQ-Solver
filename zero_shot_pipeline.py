from zero_shot.mcq_solver import ZeroShotSolver
from tqdm import tqdm
import pandas as pd
import torch
tqdm.pandas()
import mapply
mapply.init(
    n_workers=-1,     
    progressbar=True 
)

class ZeroShotPipeline:
    
    def __init__(self, test_df_path):
        
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.test_df = pd.read_csv(test_df_path)

    def get_top3(self, prompt, options, solver):

        result = solver.solve(prompt, options)
        winning_option_texts = result['labels']
        keys = ['A', 'B', 'C', 'D', 'E']
        top3_keys = []
        for text in winning_option_texts[:3]:
            original_index = options.index(text)
            top3_keys.append(keys[original_index])
            
        return " ".join(top3_keys)
    
    def run(self):
        solver = ZeroShotSolver(device_id=self.device_id)
        self.test_df['Prediction'] = self.test_df.progress_apply(
                                                    lambda x: self.get_top3(x['prompt'],[x['A'],x['B'],x['C'],x['D'],x['E']],solver), 
                                                    axis = 1)
        sub_df = self.test_df[['id','Prediction']]
        sub_df.columns = ['ID','Prediction']
        sub_df.to_csv('submission.csv',index=False)





