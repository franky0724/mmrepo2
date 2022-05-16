import argparse
import os
import os.path
import sys
import pandas as pd
import hmeq_logistic_score
import settings_b8b9553e_3617_49d7_9168_587f38e9fc52
settings_b8b9553e_3617_49d7_9168_587f38e9fc52.pickle_path = ''
parser = argparse.ArgumentParser(description='Score')
parser.add_argument('-m', dest='modelFile', help='model filename, default will be the model file hardcoded in the script')
parser.add_argument('-i', dest='scoreInputCSV', required=True, help='input filename')
parser.add_argument('-o', dest='scoreOutputCSV', required=True, help='output csv filename')
args = parser.parse_args()
model_file = args.modelFile
score_input_csv = args.scoreInputCSV
score_output_csv = args.scoreOutputCSV
if not os.path.isfile(score_input_csv):
    print('Not found input file',score_input_csv)
    sys.exit()
input_df = pd.read_csv(os.path.join(os.getcwd(), score_input_csv))
rowResult = []
for i, row in input_df.iterrows():
    rowResult.append(hmeq_logistic_score.scoreHMEQLogisticModel(row['JOB'],row['REASON'],row['CLAGE'],row['CLNO'],row['DEBTINC'],row['DELINQ'],row['DEROG'],row['NINQ'],row['YOJ']))
output_df = pd.DataFrame(rowResult, columns=['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION'])
output_df = pd.merge(input_df, output_df, how='inner', left_index=True, right_index=True)
output_df.to_csv(score_output_csv, sep=',', index=False)
