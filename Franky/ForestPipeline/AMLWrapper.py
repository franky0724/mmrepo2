import json
import os
import sys
import pandas as pd
import numpy as np

from inference_schema.schema_decorators import input_schema, output_schema

from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

sys.path.append('/opt/sas')
import maspy


def init():
    global mas

    is_astore_model = has_astore()
    ds2_file = find_pkg_score_script()

    if ds2_file is None:
       ds2_file = find_score_script('fileMetadata.json')

    if ds2_file is None:
        print('ERROR: Cannot find score code file')
        return {}

    mas = maspy.MASsf(cfgname='mascontainer')
    fix_package_line(ds2_file)

    if is_astore_model: 
        astore_file = find_file('.astore')
        if astore_file is None:
           astore_file = find_file('.sasast')
           print(astore_file)
        astore_key = get_astore_key()
        if astore_file is None:
            print('ERROR : Can not find astore file')
            return {}

        if astore_key is None:
            print('ERROR: can not find the astore_key')
            return {}

    comp = [{'name': 'ds2score', 'lang': 'ds2', 'file': ds2_file}]

    if is_astore_model:
        comp.append({'name': 'myastore', 'lang': 'astore', 'file': astore_file, 'sha1': astore_key})

    print(ds2_file)
    print('publishing...')
    ret = mas.publishComposite('ds2score', 'score', comp)

    if 'publishing apparently failed' in ret:
        print('publish failed:', ret)
        return {}

    return

def find_file(suffix):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, os.getenv('AZUREML_MODEL_DIR'))
    model_name = os.path.split(os.path.split(model_dir)[0])[1]
    full_path = os.path.join('/var/azureml-app', model_dir, model_name)

    for file in os.listdir(full_path):
        if file.endswith(suffix):
            filename = file
            return os.path.join(full_path, filename)

    return None

def find_names_by_role(filename, role):
    var_file = find_file(filename)
    if var_file is None:
        return None

    if os.path.isfile(var_file):
        with open(var_file) as f:
            json_object = json.load(f)
        names = []
        for row in json_object:
            if row['role'] == role:
                names.append(row['name'])
        if names == []:
            return None
        return names
    else:
        print('Did not find any role: ' + role + ' in file: ' + filename)
        return None

def find_score_script(filename):
    names = find_names_by_role(filename, 'score')
    if names is not None:
        return names[0]
    else:
        return None

def get_astore_key():
    meta_file = find_file('AstoreMetadata.json')
    if meta_file is None:
        meta_file = find_file('_sha1key.json')
        if meta_file is not None:
            with open(meta_file) as f:
                dt = json.load(f)
            return dt['SASTableData+_ASTOREKEY'][0]['key']
    else:
        with open(meta_file) as f:
            dt = json.load(f)
        return dt[0]['key']
    return None

def has_astore():
    meta_file = find_file('AstoreMetadata.json')
    if meta_file is None:
        return False
    else:
        return True

def fix_package_line(ds2_file):
    bad_token = 'package &'
    s = open(ds2_file).read()
    if bad_token in s:
        print("Rewriting " + ds2_file)
        s = s.replace(bad_token, 'package ')
        f = open(ds2_file, 'w')
        f.write(s)
        f.close()

def find_pkg_score_script():
    pkg_file = find_file('.ds2')
    if pkg_file is None:
        pkg_file = find_file('packagescorecode.sas')
    if pkg_file is None:
        pkg_file = find_file('pkg_score.sas')
    return pkg_file

input_sample = pd.DataFrame(data=[{'CLAGE':3.4186,'CLNO':4.7620,'LOAN':3.6582,'REASON':'jmuraxno','JOB':'xlgnruod','NINQ':1.2334,'VALUE':3.7043,'DELINQ':4.8812,'YOJ':2.5832,'DEBTINC':4.3526,'MORTDUE':2.3807,'DEROG':4.3650}])

@input_schema('data', PandasParameterType(input_sample, enforce_shape=False, enforce_column_type=False))
@output_schema(StandardPythonParameterType([{'a':1.0, 'b':2.0, 'c':1},{'a':1.1, 'b':1.9, 'c':0}]))
def run(data):
    try:
        input_df = data
        input_df = input_df.replace(r'^\s*$', np.nan, regex=True)

        inputvar_file = find_file('inputVar.json')
        if inputvar_file is not None and os.path.isfile(inputvar_file):
            print('re-order input data')
            with open(inputvar_file) as f:
                dt = json.load(f)
            names = []
            for row in dt:
                names.append(row['name'])
            df2 = input_df[names]
        else:
            print('same order as input file since there is no inputVar.json')
            df2 = input_df

        print('executing...')
        output_df = mas.execute('ds2score', 'score', df2)

        if output_df is None:
            print('Scoring failed')
            return json.dumps(dict(
                FAIL=dict(
                    msg='Scoring dataframe failed',
                    df=df2.to_dict())))

        output_df = output_df.replace(np.nan, "")
        output_df = pd.merge(input_df, output_df, how='inner', left_index=True, right_index=True)
        return output_df.to_json(orient='records')

    except Exception as e:
        print(e)
        error = str(e)
        return error
