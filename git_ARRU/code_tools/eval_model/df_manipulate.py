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
        df = pd.read_csv(
            self.file, header=0, names=self.file_col_name)
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
                    measure_numbers=None):
        self.summary_file = summary_file
        self.val_file = val_file
        self.file_col_name = file_col_name
        self.measure_numbers = measure_numbers
        #self.thre_dict = thre_dict

    def return_summary_df(self):
        if self.measure_numbers:
            df = pd.read_csv(self.summary_file, header=0,
                                names=self.file_col_name)
            df = df[df.manP!=-999]
            df_shuffle = sklearn.utils.shuffle(df)
            return df_shuffle[:self.measure_numbers]
        else:
            df = pd.read_csv(self.summary_file, header=0,
                names=self.file_col_name)
            df = df[df.manP!=-999] 
            return df

    def return_val_df(self):
        if self.measure_numbers:
            df = pd.read_csv(self.val_file, header=0,
                                names=self.file_col_name)
            df = df[df.manP!=-999]
            df_shuffle = sklearn.utils.shuffle(df)
            return df_shuffle[:self.measure_numbers]
        else:
            df = pd.read_csv(self.val_file, header=0,
                names=self.file_col_name)
            df = df[df.manP!=-999] 
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
                target_phase='P', positive_pick=thre_dict['positive_P_thre'])
        S_eva = model_evaluate(df_S, thre_dict['true_S_thre'], 
                target_phase='S', positive_pick=thre_dict['positive_S_thre'])
        
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
        P_pick_rate = len(P_TP)/len(df)
        S_pick_rate = len(S_TP)/len(df)

        MAE_P = np.fabs((P_TP['manP'] - P_TP['predP']).values)
        MAE_S = np.fabs((S_TP['manS'] - S_TP['predS']).values)
        MAPE_P = MAE_P/(P_TP['manP'].values)
        MAPE_S = MAE_S/(S_TP['manS'].values)
        p_residual = (P_TP['manP'] - P_TP['predP']).values
        s_residual = (S_TP['manS'] - S_TP['predS']).values     
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

    def grp_dataframe(self, grp_by=None, grp_range=None):
        '''
        summary_df: summary dataframe
        grp_by: group dataframe by index `ps_diff`, `raw_Psnr`, `distance`
        range: group range
        '''
        df = self.return_summary_df()
        if grp_by == 'log10(raw_Psnr)':
            # group by snr
            grp_value = np.log10(df['raw_Psnr'])
            if grp_range is None:
                grp_range = np.hstack([-np.inf, np.arange(0.5, 2.5, 0.5), np.inf])
            label = 'log10(snr)'

        elif grp_by == 'ps_diff':
            grp_value = df['ps_diff']
            if grp_range is None:
                grp_range = np.hstack([np.arange(0, 8, 2), np.inf])
            label = '|ts - tp| (sec)'

        elif grp_by == 'distance':
            grp_value = df['distance']
            if grp_range is None:
                grp_range = np.hstack([np.arange(0, 80, 20), np.inf])
            label = 'distance (km)'

        grp_idxs =  [f"grp_{i}" for i in np.arange(1, len(grp_range))]
        pd_cut = pd.cut(grp_value, bins=grp_range, labels=grp_idxs, include_lowest=True)

        grp_dfs = [df.groupby(pd_cut).get_group(i) for i in grp_idxs]
        grp_ticks = [f'({grp_range[i]:.1f}, {grp_range[i+1]:.1f}]' for i in range(len(grp_range)-1)]
        grp_ticks[-1] = f"{grp_range[-2]:.1f}+"
        grp_ticks[0] = f"{grp_range[1]:.1f}-"
        grp_cts = [len(i) for i in grp_dfs]

        return {
            'label': label,
            'grp_dfs': grp_dfs,
            'grp_ticks': grp_ticks,
            'grp_cts': grp_cts
        }

    def grp_dataframe_v2(self, grp_range_p=None, grp_range_s=None):
        '''
        grp_by: group dataframe by P/S snr separately 
        grp_range_P: group range of P
        grp_range_S
        '''
        df = self.return_summary_df()
        grp_value_p = np.log10(df['hp_Psnr'].values)
        grp_value_s = np.log10(df['hp_Ssnr'].values)
        # group by snr
        if grp_range_p is None:
            grp_range_p = np.hstack([-np.inf, np.arange(0.5, 2.0, 0.5), np.inf])
        if grp_range_s is None:
            grp_range_s = np.hstack([-np.inf, np.arange(0.4, 1.0, 0.2), np.inf])
        grp_values = [grp_value_p, grp_value_s]
        grp_ranges = [grp_range_p, grp_range_s]
        label = ['log10(P_snr)', 'log10(S_snr)']
        grp_dicts = []
        for j in range(2):
            grp_idxs =  [f"grp_{i}" for i in np.arange(1, len(grp_ranges[j]))]
            pd_cut = pd.cut(grp_values[j], bins=grp_ranges[j], labels=grp_idxs, include_lowest=True)

            grp_dfs = [df.groupby(pd_cut).get_group(i) for i in grp_idxs]
            grp_ticks = [f'({grp_ranges[j][i]:.1f}, {grp_ranges[j][i+1]:.1f}]' for i in range(len(grp_ranges[j])-1)]
            grp_ticks[-1] = f"{grp_ranges[j][-2]:.1f}+"
            grp_ticks[0] = f"{grp_ranges[j][1]:.1f}-"
            grp_cts = [len(i) for i in grp_dfs]

            grp_dicts.append( {
                'label': label[j],
                'grp_dfs': grp_dfs,
                'grp_ticks': grp_ticks,
                'grp_cts': grp_cts
            })
        return grp_dicts

class GroupEvaluate:
    def __init__(self, grp_dict, thre_dict):
        '''
        grp_dict: dict object produced by `df_manipulate.SummaryOperation.grp_dataframe
            keys: 'grp_dfs', 'grp_ticks', 'grp_cts'
        thre_dict: self defined dict or dict produced by `df_manipulate.SummaryOperation.thre_from_val_df
        '''
        self.grp_dict = grp_dict
        self.thre_dict = thre_dict

    def grp_eva_info(self):
        grp_dfs = self.grp_dict['grp_dfs']
        grp_cts = self.grp_dict['grp_cts']
        grp_ticks = self.grp_dict['grp_ticks']

        P_confusion_mtx = []
        S_confusion_mtx = []
        P_F1s = []
        S_F1s = []
        messages = []
        for i in range(len(grp_dfs)):
            df_tick = grp_ticks[i]
            df_sub = grp_dfs[i]
            df_ct = grp_cts[i]
            # successfully picked within +- 1 second with probability > 0.001 
            df_predictP = df_sub[df_sub.predP_prob!=-999]
            df_predictS = df_sub[df_sub.predS_prob!=-999]
            df_fail_predictP = df_sub[df_sub.predP_prob==-999]
            df_fail_predictS = df_sub[df_sub.predS_prob==-999]

            P_eva = model_evaluate(df_predictP, true_pick=self.thre_dict['true_P_thre'],
                                target_phase='P', positive_pick=self.thre_dict['positive_P_thre'])
            S_eva = model_evaluate(df_predictS, true_pick=self.thre_dict['true_S_thre'],
                                target_phase='S', positive_pick=self.thre_dict['positive_S_thre'])

            P_con_per = P_eva.confusion_matrix_percentage(); P_confusion_mtx.append(P_con_per)
            S_con_per = S_eva.confusion_matrix_percentage(); S_confusion_mtx.append(S_con_per)
            P_F1s.append(P_eva.f1_score())
            S_F1s.append(S_eva.f1_score())    

            message = (f"Range of {self.grp_dict['label']}:'{df_tick}' - \n"
                f"P_F1: {P_eva.f1_score():.4f}, \t\tS_F1: {S_eva.f1_score():.4f}, \n"
                f"P_precision: {P_eva.precision():.4f}, \t\tS_precision: {S_eva.precision():.4f}, \n"
                f"P_recall: {P_eva.recall():.4f}, \t\tS_recall: {S_eva.recall():.4f}, \n"
                f"P_fail_prediction_rate: {len(df_fail_predictP)}/{df_ct}, \n"
                f"S_fail_prediction_rate: {len(df_fail_predictS)}/{df_ct}\n")
            logging.info(message)
            messages.append(message)
        return {
                'messages':messages,
                'P_confusion_mtx':P_confusion_mtx,
                'S_confusion_mtx':S_confusion_mtx,
                'P_F1s':P_F1s,
                'S_F1s':S_F1s
            }   

    def grp_eva_plot(self, save=None):

        grp_ticks = self.grp_dict['grp_ticks']
        grp_cts = self.grp_dict['grp_cts']

        grp_info = self.grp_eva_info()
        P_confusion_mtx = grp_info['P_confusion_mtx']
        S_confusion_mtx = grp_info['S_confusion_mtx']
        P_F1s = grp_info['P_F1s']
        S_F1s = grp_info['S_F1s']    
        messages = grp_info['messages']

        P_tps = [i['TruePositive'] for i in P_confusion_mtx]
        P_tns = [i['TrueNegative'] for i in P_confusion_mtx]
        P_fps = [i['FalsePositive'] for i in P_confusion_mtx]
        P_fns = [i['FalseNegative'] for i in P_confusion_mtx]
        S_tps = [i['TruePositive'] for i in S_confusion_mtx]
        S_tns = [i['TrueNegative'] for i in S_confusion_mtx]
        S_fps = [i['FalsePositive'] for i in S_confusion_mtx]
        S_fns = [i['FalseNegative'] for i in S_confusion_mtx]

        fig = plt.figure(figsize=(10, 6))
        ax1, ax2, ax3 = plt.subplot(121), plt.subplot(222), plt.subplot(224)
        #== plot data distribution
        ax1.bar(grp_ticks, grp_cts, width=0.3)
        ax1.set_xlabel(self.grp_dict['label'])
        ax1.set_ylabel('Counts')
        ax1.set_title('Test data distribution (2018-2019, Taiwan)')
        #== plot P confusion matrix
        rec1 = ax2.bar(np.arange(len(P_tps))-0.4, P_tps, width=0.2, align='center', color='gray', edgecolor='k') 
        rec2 = ax2.bar(np.arange(len(P_tps))-0.2, P_tns, width=0.2, align='center', color='green', edgecolor='k')
        rec3 = ax2.bar(np.arange(len(P_tps)), P_fps, width=0.2, align='center', color='blue') 
        rec4 = ax2.bar(np.arange(len(P_tps))+0.2, P_fns, width=0.2, align='center', color='orange')
        ax2_2 = ax2.twinx()
        ax2_2.plot(np.arange(len(P_tps)), P_F1s,'-o', color='r')
        ax2_2.set_ylim(0, 1)

        ax2.set_xticks(np.arange(len(P_tps))-0.1)
        ax2.set_xticklabels('')
        ax2.set_ylabel('P phase percentage')
        ax2.set_ylim(0, 100)
        ax2.set_title(f"{self.thre_dict['positive_P_thre']:.2f} for P positive picks, {self.thre_dict['true_P_thre']:.2f} secs for P true pick\n"
                    f"{self.thre_dict['positive_S_thre']:.2f} for S positive picks, {self.thre_dict['true_S_thre']:.2f} secs for S true pick",
                    fontsize=12)
        #== plot S confusion matrix

        rec5 = ax3.bar(np.arange(len(S_tps))-0.4, S_tps, width=0.2, align='center', color='gray', edgecolor='k') 
        rec6 = ax3.bar(np.arange(len(S_tps))-0.2, S_tns, width=0.2, align='center', color='green', edgecolor='k')
        rec7 = ax3.bar(np.arange(len(S_tps)), P_fps, width=0.2, align='center', color='blue') 
        rec8 = ax3.bar(np.arange(len(S_tps))+0.2, S_fns, width=0.2, align='center', color='orange')
        ax3_2 = ax3.twinx()
        rec_f1 = ax3_2.plot(np.arange(len(S_tps)), S_F1s, '-o', color='r', label='F1_score')
        ax3_2.scatter(np.arange(len(S_tps)), S_F1s, color='r')
        ax3_2.set_ylim(0, 1)

        ax3.set_xticks(np.arange(len(S_tps))-0.1)
        ax3.legend( (rec1[0], rec2[0], rec3[0], rec4[0], rec_f1[0]), 
                ('TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative', 'F1_score'),
                bbox_to_anchor=(0.5, 1.3), ncol=2, loc='center')
        ax3.set_ylabel('S phase percentage')
        ax3.set_ylim(0, 100)
        ax3.set_xticklabels(grp_ticks)
        ax3.set_xlabel(self.grp_dict['label'])
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=100)
        plt.show()
        plt.interactive(True)
        return messages

class GroupEvaluate_snr:
    def __init__(self, grp_dicts, thre_dict):
        '''
        grp_dicts: P/S dict object produced by `df_manipulate.SummaryOperation.grp_dataframe_v2
            keys: 'label', 'grp_dfs', 'grp_ticks', 'grp_cts'
        thre_dict: self defined dict or dict produced by `df_manipulate.SummaryOperation.thre_from_val_df
        '''
        self.grp_dicts = grp_dicts
        self.thre_dict = thre_dict

    def grp_eva_info_snr(self):
        grp_eva_dicts = []
        for k in range(2):
            label = self.grp_dicts[k]['label']
            grp_dfs = self.grp_dicts[k]['grp_dfs']
            grp_cts = self.grp_dicts[k]['grp_cts']
            grp_ticks = self.grp_dicts[k]['grp_ticks']

            confusion_mtx = []
            F1s = []
            precisions = []
            recalls = []
            messages = []

            for i in range(len(grp_dfs)):
                df_tick = grp_ticks[i]
                df_sub = grp_dfs[i]
                df_ct = grp_cts[i]
                # successfully picked within +- 1 second with probability > 0.001 
                if k == 0:
                    df_predict = df_sub[df_sub.predP_prob!=-999]
                    df_fail_predict = df_sub[df_sub.predP_prob==-999]    
                    eva = model_evaluate(df_predict, true_pick=self.thre_dict['true_P_thre'],
                                        target_phase='P', positive_pick=self.thre_dict['positive_P_thre'])
                if k == 1:
                    df_predict = df_sub[df_sub.predS_prob!=-999]
                    df_fail_predict = df_sub[df_sub.predS_prob==-999]  
                    eva = model_evaluate(df_predict, true_pick=self.thre_dict['true_S_thre'],
                                        target_phase='S', positive_pick=self.thre_dict['positive_S_thre'])

                con_per = eva.confusion_matrix_percentage(); confusion_mtx.append(con_per)
                F1s.append(eva.f1_score())  
                precisions.append(eva.precision())
                recalls.append(eva.recall())

                message = (f"Range of {label}: {df_tick:15s}  "
                    f"F1 score: {eva.f1_score():.4f}, "
                    f"precision: {eva.precision():.4f}, "
                    f"recall: {eva.recall():.4f}, "
                    f"fail_prediction_rate: {len(df_fail_predict)}/{df_ct}")
                #logging.info(message)
                messages.append(message)
            grp_eva_dicts.append( {
                    'messages':messages,
                    'confusion_mtx':confusion_mtx,
                    'F1s':F1s,
                    'precisions':precisions,
                    'recalls':recalls
                })
        return grp_eva_dicts

    def grp_eva_plot_snr(self, save=None):
        grp_dicts = self.grp_dicts
        grp_eva_dicts = self.grp_eva_info_snr()

        fig = plt.figure(figsize=(10, 8))
        ax1, ax2, ax3, ax4 = plt.subplot(221), plt.subplot(223), plt.subplot(222), plt.subplot(224)
        axs = [[ax1, ax2], [ax3, ax4]]
        axs_twin = [ax1.twinx(), ax3.twinx()]
        for i in range(2):
            grp_dict = grp_dicts[i]
            grp_ticks = grp_dicts[i]['grp_ticks']
            grp_cts = np.log10(grp_dicts[i]['grp_cts'])
            #grp_cts = grp_dicts[i]['grp_cts']

            confusion_mtx = grp_eva_dicts[i]['confusion_mtx']
            F1s = grp_eva_dicts[i]['F1s']
            precisions = grp_eva_dicts[i]['precisions']
            recalls = grp_eva_dicts[i]['recalls']

            tps = [j['TruePositive'] for j in confusion_mtx]
            tns = [j['TrueNegative'] for j in confusion_mtx]
            fps = [j['FalsePositive'] for j in confusion_mtx]
            fns = [j['FalseNegative'] for j in confusion_mtx]

            #== plot data distribution
            axs[i][1].bar(np.arange(len(tps)), grp_cts, width=0.3)
            axs[i][1].set_xticks(range(len(tps)))
            axs[i][1].set_xticklabels(grp_ticks)
            axs[i][1].set_ylabel('log10(Number of samples)')
            #axs[i][1].set_ylabel('Number of samples')
            #axs[i][0].set_title('Test data distribution (2018-2019, Taiwan)')

            #== plot confusion matrix
            rec1 = axs[i][0].bar(np.arange(len(tps))-0.4, tps, width=0.2, align='center', color='gray', edgecolor='k') 
            rec2 = axs[i][0].bar(np.arange(len(tps))-0.2, tns, width=0.2, align='center', color='green', edgecolor='k')
            rec3 = axs[i][0].bar(np.arange(len(tps)), fps, width=0.2, align='center', color='blue') 
            rec4 = axs[i][0].bar(np.arange(len(tps))+0.2, fns, width=0.2, align='center', color='orange')
            axs[i][0].set_xticks(range(len(tps)))
            axs[i][0].set_xticklabels(grp_ticks)
            rec_f1 = axs_twin[i].plot(np.arange(len(tps)), F1s,'-o', color='r')
            rec_precision = axs_twin[i].plot(np.arange(len(tps)), precisions,'-o', color='brown')
            rec_recall = axs_twin[i].plot(np.arange(len(tps)), recalls,'-o', color='k')
            axs_twin[i].set_ylim(0, 1)

            axs[i][0].set_xticklabels(grp_ticks, rotation=30, fontsize=10)
            axs[i][1].set_xticklabels(grp_ticks, rotation=30, fontsize=10)
            axs[i][1].set_xlabel(grp_dict['label'])
            axs[i][0].set_ylabel('Confusion matrix component')
            axs[i][0].set_ylim(0, 100)
            #axs[i][0].axes.xaxis.set_visible(False)

            #== plot S confusion matrix
        axs[i][0].set_ylabel('')
        axs[i][1].set_ylabel('')
        axs[i][0].set_yticklabels('')
        ax1.set_xticks(np.arange(len(tps))-0.1)
        ax1.legend( (rec1[0], rec2[0], rec3[0], rec4[0], rec_f1[0], rec_precision[0], rec_recall[0]), 
                        ('TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative',
                         'F1-score', 'Precision', 'Recall'),
                        bbox_to_anchor=(1.1, 1.20), ncol=4, loc='center')
        ax1.set_title('P phase')
        ax3.set_title('S phase')
        if save:
            plt.savefig(save, dpi=100)
        plt.show()
        plt.interactive(True)
        return fig