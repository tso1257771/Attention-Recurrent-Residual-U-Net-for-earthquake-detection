import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=12
plt.rcParams['font.family']='Helvetica'
from sklearn.metrics import auc
# precision: P = Tp / (Tp+Fp)
# recall:    R = Tp / (Tp+Fn)
# F1_score = 2 * (P*R)/(P+R)
# accuracy = (Tp+Tn) /(Tp+Tn+Fp+Fn) #observe true pick
# false positive rate = Fp/(Fp+Tn)
# true negative rate = Tn/(Fn+Tn)

class model_evaluate:
    def __init__(self, df_predict, true_pick, target_phase, positive_pick):
        self.df_length = len(df_predict)
        self.df_predict = df_predict
        self.true_pick = true_pick
        self.positive_pick = positive_pick
        self.target_phase = target_phase

        if self.target_phase.lower() == 'p':
            self.df_fail_idx = np.array(self.df_predict.predP == -999)

            self.df_TruePositive_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manP-self.df_predict.predP) < self.true_pick),
                np.array(self.df_predict.predP_prob > self.positive_pick ))

            self.df_TrueNegative_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manP-self.df_predict.predP) > self.true_pick),
                np.array(self.df_predict.predP_prob < self.positive_pick ))

            self.df_FalsePositive_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manP-self.df_predict.predP) > self.true_pick),
                np.array(self.df_predict.predP_prob > self.positive_pick ))

            self.df_FalseNegative_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manP-self.df_predict.predP) < self.true_pick),
                np.array(self.df_predict.predP_prob < self.positive_pick ))

            self.df_TruePositive = self.df_predict[self.df_TruePositive_idx]
            self.df_TrueNegative = self.df_predict[self.df_TrueNegative_idx]
            self.df_FalsePositive = self.df_predict[self.df_FalsePositive_idx]
            self.df_FalseNegative = self.df_predict[self.df_FalseNegative_idx]

        elif self.target_phase.lower() == 's':
            self.df_fail_idx = np.array(self.df_predict.predS == -999)

            self.df_TruePositive_idx =  np.logical_and(
                np.array(np.fabs(self.df_predict.manS-self.df_predict.predS) < self.true_pick),
                np.array(self.df_predict.predS_prob > self.positive_pick))

            self.df_TrueNegative_idx =  np.logical_and(
                np.array(np.fabs(self.df_predict.manS-self.df_predict.predS) > self.true_pick),
                np.array(self.df_predict.predS_prob < self.positive_pick))
                
            self.df_FalsePositive_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manS-self.df_predict.predS) > self.true_pick),
                np.array(self.df_predict.predS_prob > self.positive_pick))

            self.df_FalseNegative_idx = np.logical_and(
                np.array(np.fabs(self.df_predict.manS-self.df_predict.predS) < self.true_pick),
                np.array(self.df_predict.predS_prob < self.positive_pick))

            self.df_TruePositive = self.df_predict[self.df_TruePositive_idx]
            self.df_TrueNegative = self.df_predict[self.df_TrueNegative_idx]
            self.df_FalsePositive = self.df_predict[self.df_FalsePositive_idx]
            self.df_FalseNegative = self.df_predict[self.df_FalseNegative_idx]

        self.TruePositive = len(self.df_TruePositive)
        self.TrueNegative = len(self.df_TrueNegative) + len(df_predict[self.df_fail_idx])
        self.FalsePositive = len(self.df_FalsePositive)
        self.FalseNegative = len(self.df_FalseNegative)
        
    def confusion_matrix(self):
        return {
        'TruePositive' : self.TruePositive,
        'TrueNegative' : self.TrueNegative,
        'FalsePositive' :self.FalsePositive,
        'FalseNegative' :self.FalseNegative 
        }

    def confusion_matrix_percentage(self):
        return {
        'TruePositive' : self.TruePositive/self.df_length*100,
        'TrueNegative' : self.TrueNegative/self.df_length*100,
        'FalsePositive' :self.FalsePositive/self.df_length*100,
        'FalseNegative' :self.FalseNegative/self.df_length*100 
        }

    def TrueNegative_rate(self):
        try:
            return self.TrueNegative/(self.TrueNegative+self.FalseNegative)
        except:
            return 0

    def FalsePositive_rate(self):
        try:
            return self.FalsePositive/(self.FalsePositive+self.TrueNegative)
        except:
            return 0

    def precision(self):
        try:
            return self.TruePositive/(self.TruePositive+self.FalsePositive)
        except:
            return 0

    def recall(self):
        try:
            return self.TruePositive/(self.TruePositive+self.FalseNegative)
        except:
            return 0
    
    def accuracy(self):
        try:
            return (self.TruePositive+self.TrueNegative)/ \
                      (self.TruePositive+self.TrueNegative+self.FalsePositive+self.FalseNegative)
        except:
            return 0
    
    def f1_score(self):
        try:
            return 2*(self.precision()*self.recall())/(self.precision()+self.recall())
        except:
            return 0
    '''
    def AUROC(self, prob_thre=np.arange(0.05, 1.05, 0.05)):
        #prob_thre = np.arange(0.05, 1.05, 0.05)
        fprs = [model_evaluate(self.df_predict, self.true_pick, self.target_phase, positive_pick=i).FalsePositive_rate()
                                                                 for i in prob_thre]
        recalls = [model_evaluate(self.df_predict, self.true_pick, self.target_phase, positive_pick=i).recall()
                                                                 for i in prob_thre]
        AUC = auc(fprs, recalls)
        return {
            'positive_values':prob_thre, 
            'FalsePositive_rates':fprs, 
            "recalls":recalls, 
            "AUC":AUC
            }
    '''
    def AUCs(self, prob_thre=np.arange(0.0, 1.05, 0.05)):
        #prob_thre = np.arange(0.05, 1.05, 0.05)
        fprs        = np.array([model_evaluate(self.df_predict, self.true_pick, self.target_phase,
                                positive_pick=i).FalsePositive_rate() for i in prob_thre])
        precisions  = np.array([model_evaluate(self.df_predict, self.true_pick, self.target_phase, 
                                positive_pick=i).precision() for i in prob_thre])
        recalls     = np.array([model_evaluate(self.df_predict, self.true_pick, self.target_phase, 
                                positive_pick=i).recall() for i in prob_thre])
        f1s         = np.array([model_evaluate(self.df_predict, self.true_pick, self.target_phase, 
                                positive_pick=i).f1_score() for i in prob_thre])  

        '''
        # delete zero point
        zero_pt = np.where(np.logical_and(precisions==0, recalls==0, fprs==0))[0]

        fprs = np.delete(fprs, zero_pt)
        precisions = np.delete(precisions, zero_pt)
        recalls = np.delete(recalls, zero_pt)
        f1s = np.delete(f1s, zero_pt)
        prob_thre = np.delete(prob_thre, zero_pt)
        '''
        AUROC = auc(fprs, recalls)
        pr_AUC = auc(recalls, precisions)                                                       
        f1_auc = auc(prob_thre, f1s)
        
        return {
            "positive_values":prob_thre, 
            "FalsePositive_rate":fprs,
            "recalls":recalls, 
            "precisions":precisions,
            "f1s":f1s,
            "AUROC":AUROC, 
            "pr_AUC":pr_AUC,
            "f1_AUC":f1_auc
            }
    

if __name__ == '__main__':
    pass