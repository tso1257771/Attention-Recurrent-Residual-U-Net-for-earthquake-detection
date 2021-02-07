import os
import logging
import sys
import numpy as np
sys.path.append('../')
from matplotlib import interactive
from code_tools.eval_model.evaluate_performance import model_evaluate
from code_tools.eval_model.df_manipulate import SummaryOperation
from code_tools.eval_model.df_manipulate import GroupEvaluate_snr
from code_tools.eval_model.df_manipulate import DetectionEval
from code_tools.data_utils import gen_tar_func
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s : %(asctime)s : %(message)s')

mdl_hdr = 'ARRU_detect/EQT_detect_20s'
mdl_test = mdl_hdr + '_on_STEAD'
basepath = f'../s03_test_predict/{mdl_test}'
save_fig = True
plt_save = f'../s04_model_evaluate/{mdl_test}_snr.png'
out_sum = f'../s04_model_evaluate/{mdl_test}.txt'
summary_file = os.path.join(basepath, 'summary.txt')
val_file = os.path.join(basepath+'_val', 'summary.txt')
file_col_name = ['evid', 'sta', 'chn', 'ps_diff', 'manP', 'manS', 'predP', 
    'predP_prob', 'predS', 'predS_prob', 'distance','hp_Psnr', 'hp_Ssnr',
    'real_type', 'pred_type', 'eq_prob']

summary = SummaryOperation(summary_file, val_file, file_col_name)
thre_dict = {'true_P_thre': 0.5, 'true_S_thre': 0.5, 
                'positive_P_thre': 0.3, 'positive_S_thre': 0.3}

test_DectEval = DetectionEval(summary_file, file_col_name)
val_DectEval = DetectionEval(val_file, file_col_name)
detection_test = test_DectEval.eval_mtx()
detection_val = val_DectEval.eval_mtx()
# overall evaluation
P_score_test, S_score_test = summary.overall_eval(
    thre_dict=thre_dict, prob_thre=np.arange(0., 1.01, 0.01))
P_score_val, S_score_val = summary.overall_eval(
    thre_dict=thre_dict, return_val=True, prob_thre=np.arange(0., 1.01, 0.01))
overall_message = (
    f"              \t  {'Test data':>6s}\t{'Validation data':>6s}\n"
    f"-------------------------------------------------------------------------------------------------------\n"
    f"Detection\n"
    f"-------------------------------------------------------------------------------------------------------\n"
    f"Precision         {detection_test['precision']:>2.4f}\t{detection_val['precision']:>2.4f}\n"
    f"Recall            {detection_test['recall']:>2.4f}\t{detection_val['recall']:>2.4f}\n"
    f"F1 score          {detection_test['f1_score']:>2.4f}\t{detection_val['f1_score']:>2.4f}\n"    
    f"-------------------------------------------------------------------------------------------------------\n"
    f"P performance\n"
    f"-------------------------------------------------------------------------------------------------------\n"
    f"Picking rate      {P_score_test['pick_rate']:>2.4f}\t{P_score_val['pick_rate']:>2.4f}\n"
    f"Residual mean (s) {P_score_test['residual'].mean():>2.4f}\t{P_score_val['residual'].mean():>2.4f}\n"
    f"Residual STD (s)  {P_score_test['residual'].std():>2.4f}\t{P_score_val['residual'].std():>2.4f}\n"
    f"Precision         {P_score_test['precision']:>2.4f}\t{P_score_val['precision']:>2.4f}\n"
    f"Recall            {P_score_test['recall']:>2.4f}\t{P_score_val['recall']:>2.4f}\n"
    f"F1 score          {P_score_test['f1_score']:>2.4f}\t{P_score_val['f1_score']:>2.4f}\n"
    f"MAE (s)           {P_score_test['MAE'].mean():>2.4f}\t{P_score_val['MAE'].mean():>2.4f}\n"
    f"MAPE (s)          {P_score_test['MAPE'].mean():>2.4f}\t{P_score_val['MAPE'].mean():>2.4f}\n" 
    f"-------------------------------------------------------------------------------------------------------\n"
    f"S performance\n"
    f"-------------------------------------------------------------------------------------------------------\n"
    f"Picking rate      {S_score_test['pick_rate']:>2.4f}\t{S_score_val['pick_rate']:>2.4f}\n"
    f"Residual mean (s) {S_score_test['residual'].mean():>2.4f}\t{S_score_val['residual'].mean():>2.4f}\n"
    f"Residual STD (s)  {S_score_test['residual'].std():>2.4f}\t{S_score_val['residual'].std():>2.4f}\n"
    f"Precision         {S_score_test['precision']:>2.4f}\t{S_score_val['precision']:>2.4f}\n"
    f"Recall            {S_score_test['recall']:>2.4f}\t{S_score_val['recall']:>2.4f}\n"
    f"F1 score          {S_score_test['f1_score']:>2.4f}\t{S_score_val['f1_score']:>2.4f}\n"
    f"MAE (s)           {S_score_test['MAE'].mean():>2.4f}\t{S_score_val['MAE'].mean():>2.4f}\n"
    f"MAPE (s)          {S_score_test['MAPE'].mean():>2.4f}\t{S_score_val['MAPE'].mean():>2.4f}\n"      
    f"-------------------------------------------------------------------------------------------------------\n"
    )

f=open(out_sum, 'w+')
print("Overall evaluation\n", file=f)
print(overall_message, file=f)
f.close()

r = open(out_sum, 'r')
len_sum = str(len(r.readlines()))
r.close()
os.system(f"head  {out_sum} -n {len_sum}")
logging.info(f"Saved to {out_sum}")

