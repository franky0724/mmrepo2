import os
import os.path
import sys
import pandas as pd
import numpy as np
import subprocess
import pkg_resources
import json

from inference_schema.schema_decorators import input_schema, output_schema

from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

model_dir = os.getenv('AZUREML_MODEL_DIR')
model_name = os.path.split(os.path.split(model_dir)[0])[1]
full_path = os.path.join("/var/azureml-app", model_dir, model_name)
sys.path.append(full_path)

import settings_b8b9553e_3617_49d7_9168_587f38e9fc52
import hmeq_logistic_score

def init():
    base_dir = "/var/azureml-app"
    requirements_json = "requirements.json"
    requirements_json_path = os.path.relpath(full_path, base_dir)
    requirements_json_file = os.path.join(requirements_json_path, requirements_json)

    required = set()
    if os.path.isfile(requirements_json_file):
        with open(requirements_json_file) as f:
            data = json.load(f)

            for package in data:
                command = package['command'].split()
                required.add(command[len(command)-1])

    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    print('missing packages: ', missing)

    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

input_sample = pd.DataFrame(data=[{'CLNO':4.4493,'DEROG':1.2389,'JOB':'yccphubd','CLAGE':2.7942,'DELINQ':4.4206,'REASON':'xevtqttr','YOJ':4.7110,'DEBTINC':3.7193,'NINQ':1.8146}])

@input_schema('data', PandasParameterType(input_sample, enforce_shape=False, enforce_column_type=False))
@output_schema(StandardPythonParameterType([{'a':1.0, 'b':2.0, 'c':1},{'a':1.1, 'b':1.9, 'c':0}]))
def run(data):
    settings_b8b9553e_3617_49d7_9168_587f38e9fc52.pickle_path = full_path + "/"

    input_df = data
    input_df = input_df.replace(r'^\s*$', np.nan, regex=True)

    row_result = []
    for i, row in input_df.iterrows():
        row_result.append(hmeq_logistic_score.scoreHMEQLogisticModel(row['JOB'],row['REASON'],row['CLAGE'],row['CLNO'],row['DEBTINC'],row['DELINQ'],row['DEROG'],row['NINQ'],row['YOJ']))
    output_df = pd.DataFrame(row_result, columns=['EM_EVENTPROBABILITY','EM_CLASSIFICATION'])
    output_df = pd.merge(input_df, output_df, how='inner', left_index=True, right_index=True)

    output_df = output_df.replace(np.nan, "")

    if output_df is None:
        return json.dumps(dict(
            FAIL=dict(
                msg="Scoring dataframe failed",
                df=input_df.to_dict())))

    return output_df.to_json(orient="records")