import logging
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size']=12
plt.rcParams['font.family']='Helvetica'
from sklearn.metrics import auc
from .evaluate_performance import model_evaluate
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

class DetectionEval:
    '''Evaluate detection performance from 
        earthquake label and noise label
    '''
    def __init__(self, df_file, file_col_name):
        self.file = df_file
        self.file_col_name = file_col_name

    def return_df(self):
        df = pd.read_table(
            self.file, header=0, names=self.file_col_name,
                delimiter=',', sep='\s+')
        return df
    
    def __pred_label_strip(self):
        '''Remove the whitespace in prediction docs
        '''
        df = self.return_df()
        real_type = np.char.strip(
            df.real_type.values.astype(str))
        pred_type = np.char.strip(
            df.pred_type.values.astype(str))
        return real_type, pred_type

    def __replace_type_as_int(self, mask_array):
        '''Assign label `earthquake` as 1
                        `noise` as 0
        ''' 
        mask_array = np.char.replace(mask_array, 
            'earthquake', str(1))
        mask_array = np.char.replace(mask_array, 
            'noise', str(0))
        return mask_array.astype(int)

    def transform_labels(self):
        
        real_type, pred_type = self.__pred_label_strip()
        real_type = self.__replace_type_as_int(real_type)
        pred_type = self.__replace_type_as_int(pred_type)

        return real_type, pred_type
        
    def confusion_comp(self, real_type=None, pred_type=None):
        '''Building confusion matrix
        '''
        if np.logical_and(real_type==None, pred_type==None):
            real_type, pred_type = self.transform_labels()
        assert len(real_type) == len(pred_type)

        # TruePositive: pred = EQ, real = EQ
        tp = np.where(np.logical_and(
            (real_type-pred_type)==0, real_type==1))[0]
        # TrueNegative: pred = noise, real = noise
        tn = np.where(np.logical_and(
            (real_type-pred_type)==0, real_type==0))[0]
        # FalseNegative: pred = noise, real = EQ
        fn = np.where(real_type-pred_type==1)[0]
        # FalsePositve: pred = EQ, real = noise
        fp = np.where(real_type-pred_type==-1)[0]
        assert len(tp)+len(tn)+len(fn)+len(fp)==len(pred_type)
        return tp, tn, fn, fp
    
    def eval_mtx(self):
        '''Evaluation matrix
        '''
        tp, tn, fn, fp = self.confusion_comp()
        precision = len(tp) / (len(tp)+len(fp))
        recall = len(tp) / (len(tp)+len(fn))
        f1 = 2*precision*recall/(precision+recall)
        detect_score = {
            'precision':precision, 
            'recall':recall, 
            'f1_score':f1}
        return detect_score


def arpicker_overall_eval(df, thre_dict=None, return_val=False, val_df=None):
    '''
    return f1 score of across overall prediction results
    '''
    if return_val:
        df = val_df

    df_P = df[df['predP']!=-999]
    df_S = df[df['predS']!=-999]
        
    P_pick_rate = len(df_P)/len(df)
    S_pick_rate = len(df_S)/len(df)
            
    MAE_P = np.fabs((df_P['manP'] - df_P['predP']).values)
    MAE_S = np.fabs((df_S['manS'] - df_S['predS']).values)
    MAPE_P = MAE_P/(df_P['manP'].values)
    MAPE_S = MAE_S/(df_S['manS'].values)
    p_residual = (df_P['manP'] - df_P['predP']).values
    s_residual = (df_S['manS'] - df_S['predS']).values     
    # Positive/Negative picks thresholds    
    #df_P_pick = df_P[df_P.predP>0]
    #df_S_pick = df_S[df_S.predS>0]
    P_score = {'pick_rate':P_pick_rate,
                'residual':p_residual, 
                'MAE':MAE_P, 'MAPE':MAPE_P}
    S_score = {'pick_rate':S_pick_rate,
                'residual':s_residual, 
                'MAE':MAE_S, 'MAPE':MAPE_S}
    return P_score, S_score

class SummaryOperation:
    def __init__(self, summary_file, val_file, file_col_name,
                    measure_numbers=None, dataset_type=None):
        self.summary_file = summary_file
        self.val_file = val_file
        self.file_col_name = file_col_name
        self.measure_numbers = measure_numbers
        self.dataset_type = dataset_type # "STEAD"
        #self.thre_dict = thre_dict

    def return_summary_df(self):
        if self.measure_numbers:
            df = pd.read_table(
            self.summary_file, header=0, names=self.file_col_name,
                delimiter=',', sep='\s+')
            df = df[df.manP > -900]
            df_shuffle = sklearn.utils.shuffle(df)
            return df_shuffle[:self.measure_numbers]
        else:
            df = pd.read_table(
            self.summary_file, header=0, names=self.file_col_name,
                delimiter=',', sep='\s+')
            df = df[df.manP > -900] 
            return df

    def return_val_df(self):
        if self.measure_numbers:
            df = pd.read_csv(self.val_file, header=0,
                                names=self.file_col_name)
            df = df[df.manP > -900]
            df_shuffle = sklearn.utils.shuffle(df)
            return df_shuffle[:self.measure_numbers]
        else:
            df = pd.read_csv(self.val_file, header=0,
                names=self.file_col_name)
            df = df[df.manP > -900] 
            return df

    def summary_df_info(self):
        '''
        return predicted probabilities, 
        residual between predicted and labeled phase
        '''
        sum_df = self.return_summary_df()
        # True/False picks thresholds
        valid_p_idx = sum_df['predP_prob']>0
        valid_s_idx = sum_df['predS_prob']>0
        p_residual = np.fabs((sum_df['manP'][valid_p_idx] -\
                        sum_df['predP'][valid_p_idx]).values)
        s_residual = np.fabs((sum_df['manS'][valid_s_idx] - \
                        sum_df['predS'][valid_s_idx]).values)
        # Positive/Negative picks thresholds    
        p_prob = sum_df['predP_prob'][valid_p_idx]
        s_prob = sum_df['predS_prob'][valid_s_idx]
        return {
            'p_residual':p_residual,
            's_residual':s_residual,
            'p_prob':p_prob,
            's_prob':s_prob
        }

    def thre_from_val_df(self):
        '''
        val_file: validation dataset prediction summary file

        true picks threshold = residual.mean() + residual.std()
        false picks threshold = prob.mean() - prob.std()
        return: thre_true_P, thre_true_S, thre_positive_P, thre_positive_S
        '''
        val_df = self.return_val_df()
        # True/False picks thresholds
        valid_p_idx = val_df['predP_prob']>0
        valid_s_idx = val_df['predS_prob']>0
        p_residual = np.fabs((val_df['manP'][valid_p_idx] -\
                    val_df['predP'][valid_p_idx]).values)
        s_residual = np.fabs((val_df['manS'][valid_s_idx] -\
                    val_df['predS'][valid_s_idx]).values)
        # Positive/Negative picks thresholds    
        p_prob = val_df['predP_prob'][valid_p_idx]
        s_prob = val_df['predS_prob'][valid_s_idx]
        return {
            'true_P_thre':p_residual.mean()+p_residual.std(), 
            'true_S_thre':s_residual.mean()+s_residual.std(),
            'positive_P_thre':p_prob.mean()-p_prob.std(),
            'positive_S_thre':s_prob.mean()-s_prob.std()
        }

    def overall_eval(self, thre_dict=None, return_val=False,
             evaluate_AUCS=False, 
             prob_thre = np.arange(0.0, 1.00, 0.01)):
        '''
        return f1 score of across overall prediction results
        '''
        if return_val:
            df = self.return_val_df()
        else:
            df = self.return_summary_df()
        df_P = df.copy()
        df_S = df.copy()

        if not thre_dict:
            thre_dict = self.thre_from_val_df()
        P_eva = model_evaluate(df_P, thre_dict['true_P_thre'], 
                target_phase='P', positive_pick=thre_dict['positive_P_thre'],
                dataset_type=self.dataset_type)
        S_eva = model_evaluate(df_S, thre_dict['true_S_thre'], 
                target_phase='S', positive_pick=thre_dict['positive_S_thre'],
                dataset_type=self.dataset_type)
        
        # Estimating AUCs
        if evaluate_AUCS:
            logging.info('Estimating AUC ...')
            P_aucs = P_eva.AUCs(prob_thre=prob_thre)
            S_aucs = S_eva.AUCs(prob_thre=prob_thre)
        elif not evaluate_AUCS:
            P_aucs =   {
                "positive_values":0, 
                "FalsePositive_rate":0,
                "recalls":0, 
                "precisions":0,
                "f1s":0,
                "AUROC":0, 
                "pr_AUC":0,
                "f1_AUC":0
                }
            S_aucs = P_aucs
        # True positives estimate
        P_TP = P_eva.df_TruePositive
        S_TP = S_eva.df_TruePositive
        P_pick_rate = P_eva.picking_rate()
        S_pick_rate = S_eva.picking_rate()

        MAE_P = np.fabs((P_TP['manP'].values - P_TP['predP'].values))
        MAE_S = np.fabs((S_TP['manS'].values - S_TP['predS'].values))
        MAPE_P = MAE_P/(P_TP['manP'].values)
        MAPE_S = MAE_S/(S_TP['manS'].values)        
        p_residual = (P_TP['manP'].values - P_TP['predP'].values)
        s_residual = (S_TP['manS'].values - S_TP['predS'].values)     
        # Positive/Negative picks thresholds    
        #df_P_pick = df_P[df_P.predP>0]
        #df_S_pick = df_S[df_S.predS>0]
        p_prob = df_P['predP_prob']
        s_prob = df_S['predS_prob']
        P_score = {'f1_score':P_eva.f1_score(), 'precision':P_eva.precision(),
                     'recall':P_eva.recall(), 'pick_rate':P_pick_rate,
                     'residual':p_residual, 'prob':p_prob, 'AUC':P_aucs,
                     'MAE':MAE_P, 'MAPE':MAPE_P}
        S_score = {'f1_score':S_eva.f1_score(), 'precision':S_eva.precision(), 
                     'recall':S_eva.recall(), 'pick_rate':S_pick_rate,
                     'residual':s_residual, 'prob':s_prob, 'AUC':S_aucs,
                     'MAE':MAE_S, 'MAPE':MAPE_S}
        return P_score, S_score
