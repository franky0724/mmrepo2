import scipy.special as special
import math

def scoreHMEQLogisticModel (JOB, REASON, CLAGE, CLNO, DEBTINC, DELINQ, DEROG, NINQ, YOJ):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    # Here are the parameter estimates for the Prob(BAD=1)/Prob(BAD=0) logit
    _paramEst = {'const':          -5.210983100988667,
                 'JOB_Mgr':        -0.25138266842783563,
                 'JOB_Office':     -0.8275612145209014,
                 'JOB_Other':      -0.24791100953872106,
                 'JOB_ProfExe':    -0.021304445941257594,
                 'JOB_Sales':       0.9002389027056661,
                 'REASON_DebtCon': -0.07612559422050401,
                 'CLAGE':          -0.0034528099235703147,
                 'CLNO':           -0.028295949751449456,
                 'DEBTINC':         0.10372743626328933,
                 'DELINQ':          0.8573938610548035,
                 'DEROG':           0.6616300801649797,
                 'NINQ':            0.14409050625224126,
                 'YOJ':            -0.005793109610043724}

    # Threshold for the misclassification error (BAD: 0-No, 1-Yes)
    _threshPredProb = 0.08941485864562787

    # Calculate the X*beta
    # The Intercept term
    _xbeta = _paramEst['const']

    # The JOB categorical feature (Self is the reference category)
    # Impute to Other if JOB is missing or blank
    try:
        _cStr = JOB.strip()
    except AttributeError:
        _cStr = 'Other'

    if (_cStr == 'Mgr'):
        _xbeta += _paramEst['JOB_Mgr']
    elif (_cStr == 'Office'):
        _xbeta += _paramEst['JOB_Office']
    elif (_cStr == 'Other'):
        _xbeta += _paramEst['JOB_Other']
    elif (_cStr == 'ProfExe'):
        _xbeta += _paramEst['JOB_ProfExe']
    elif (_cStr == 'Sales'):
        _xbeta += _paramEst['JOB_Sales']
    elif (_cStr == 'Self'):
        _xbeta += 0.0
    else:
        _xbeta += _paramEst['JOB_Other']

    # The REASON categorical feature (HomeImp is the reference category)
    # Impute to DebtCon if REASON is missing or blank
    try:
        _cStr = REASON.strip()
    except AttributeError:
        _cStr = 'DebtCon'

    if (_cStr == 'DebtCon'):
        _xbeta += _paramEst['REASON_DebtCon']
    elif (_cStr == 'HomeImp'):
        _xbeta += 0.0
    else:
        _xbeta += _paramEst['REASON_DebtCon']

    # Impute missing values with means or modes from training data 
    try:
        if math.isnan(CLAGE):
            CLAGE = 173.46666666666600
    except TypeError:
        CLAGE = 173.46666666666600
    
    try:
        if math.isnan(CLNO):
            CLNO = 20.0
    except TypeError:
        CLNO = 20.0

    try:
        if math.isnan(DEBTINC):
            DEBTINC = 34.81826181858690
    except TypeError:
        DEBTINC = 34.81826181858690

    try:
        if math.isnan(DELINQ):
            DELINQ = 0.45
    except TypeError:
        DELINQ = 0.45

    try:
        if math.isnan(DEROG):
            DEROG = 0.0
    except TypeError:
        DEROG = 0.0

    try:
        if math.isnan(NINQ):
            NINQ = 0.0
    except TypeError:
        NINQ = 0.0

    try:
        if math.isnan(YOJ):
            YOJ = 7.0
    except TypeError:
        YOJ = 7.0

    # The CLAGE continuous feature
    _xbeta += _paramEst['CLAGE'] * CLAGE

    # The CLNO continuous feature
    _xbeta += _paramEst['CLNO'] * CLNO

    # The DEBTINC continuous feature
    _xbeta += _paramEst['DEBTINC'] * DEBTINC

    # The DELINQ continuous feature
    _xbeta += _paramEst['DELINQ'] * DELINQ

    # The DEROG continuous feature
    _xbeta += _paramEst['DEROG'] * DEROG

    # The NINQ continuous feature
    _xbeta += _paramEst['NINQ'] * NINQ

    # The YOJ continuous feature
    _xbeta += _paramEst['YOJ'] * YOJ

    # The predicted probability that BAD = 1
    # The expit() function handles floating point over- and under-flow better
    # Beware that expit() returns a numpy.float64 which PyMAS does not know
    EM_EVENTPROBABILITY = float(special.expit(_xbeta))

    if (EM_EVENTPROBABILITY >= _threshPredProb):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0'

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)
