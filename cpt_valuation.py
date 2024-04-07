import math

PRELEC_LOOKUP = {}

#Takes tuples of form (outcome,outcome_prob)
def evaluateProspectVals(params,prospects):
    prospect_vals = []
    # print("params: ")
    # print(params)
    if (len(params) == 3):
        #alpha = beta
        #gamamplys = gammaminus
        #lambda = 1
        #dont use theta
        full_params = (params[0],params[0],1.,params[1],params[1])
        # print("full_params: ")
        # print(full_params)
        for prospect in prospects:
            prospect_vals.append(evaluateSingleProspectValue(prospect,full_params))
    else:
        for prospect in prospects:
            prospect_vals.append(evaluateSingleProspectValue(prospect,params))
            
    return prospect_vals    

powerValDict = dict()
def powerValue(x,alpha,beta,lamb):
    key = (x,alpha,beta,lamb)
    # print("x: ")
    # print(x)

    if (key  not in powerValDict):
        if (x >= 0.0):
            arg= math.pow(float(x),alpha)
        else:
            arg= -1*lamb*math.pow(-float(x),beta)
        powerValDict[key] = arg
        return arg
    else:
        return powerValDict[key]

def prelecWeight(gamma,p):
    if (abs(p - 1.0) < 0.0001):
        return 1.0
    if (abs(p - 0.0) < 0.0001):
        return 0.0
    if (p==1.0 or p==0.0):
        return p
    # print(gamma)
    if ((p,gamma) not in PRELEC_LOOKUP):
        # print("here:")
        # print(p)
        # print(gamma)
        try:
            val = math.exp(-1*(math.pow(math.log(1.0/p),gamma)))
        except:
            print ("Error ",sys.exc_info(),"in calculating prelec weight (p,gamma)",p,gamma)
            assert(0)
        PRELEC_LOOKUP[(p,gamma)] = val
        return val
    return PRELEC_LOOKUP[(p,gamma)]

"""
Translate (valA-valB) into probability.
Limit exponent calculation to exp(-300) min or exp(300) max
"""
def logitError(valA,valB,sensitivity):
    expval =-1*sensitivity*(valA-valB)
    #limit to (-10,+10)
    expval = min(max(expval,-300.0),300.0)
    result = 1.0/(1.0+math.exp(expval))
    return result

def evaluateSingleProspectValue(prospect,param):

    (alpha,beta,lamb,gammaGain,gammaLoss) = param
    prospectVal = 0.0
    # print("prospect: ")
    # print(prospect)
    for outcome,prob in prospect:
        # print(prospect)
        # print("outcome:")
        # print(outcome)
        # print(alpha)
        # print(beta)
        # print(lamb)
        if prob < 0:
            prob = 0.00001

        if prob > 1:
            prob = 0.99999

        tempVal = powerValue(outcome,alpha,beta,lamb)
        if (outcome > 0.0):
            # print("gammaGain: ")
            # print(gammaGain)
            # print("prob: ")
            # print(prob)
            tempWeight = prelecWeight(gammaGain,prob)
        else:
            tempWeight = prelecWeight(gammaLoss,prob)
        prospectVal = prospectVal + tempVal*tempWeight
    return prospectVal
