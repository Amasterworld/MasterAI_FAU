'''
    Your assignment is to implement the following functions,
    i.e. replace the `raise NotImplementedError()` with code that returns the right answer.

    Do not use any libraries!

    We will partly automate the evaluation so please make sure you
    that you don't change the signature of functions.
    Also please use a recent python version (at least version 3.6).
    
    You may call functions that you've implemented in other functions.
    You may also implement helper functions.

    Understanding the arguments:
        Omega:  the sample space, represented as a list
        P:      the probability function (P : Omega ⟶ float)
        ValX:   the possible values of a random variable X, represented as a list
        VarX:   a random variable (VarX : Omega ⟶ ValX), here represented as a function
        x:      a concrete value for VarX (x ∈ ValX)
        EventA: an event a, represented as list of pairs [(VarA1, a1), (VarA2, a2), ...]
                representing the event a := (VarA1=a1) ∧ (VarA2=a2) ∧ …

    Example code: given Omega, P, VarX, x
        w = Omega[0]       # pick the first sample (note that the order is meaningless)
        print(P(w))        # print the probability of that sample
        if VarX(w) == x:   # compute the value of the random variable for this sample and compare it to x
            print('X = x holds for this sample')
        else:
            print('X = x doesn't hold for this sample')

    Example call:
        def isEven(n):
            if n%2 == 0:
                return 'yes'
            else:
                return 'no'
        def probfunction(n):
            return 1/6    # fair die
        print('P(isEven = yes) for a fair die:')
        print(unconditional_probability([1,2,3,4,5,6], probfunction, isEven, 'yes'))

        for omega in omega:
            if isEven(omega) == ' yes':
'''


def unconditional_probability(Omega, P, VarX, x):
    ''' P(VarX = x)
        Hint: We marginalize all other variables than VarX and compute
          P(VarX = x) = Sum_{all possible values of all other RVs} P(those values, VarX = x)
          Note that the list of all events is represented by the list Omega,
          so the list of all possible values of all other RVs is
          the sublist of Omega containing those elements where VarX = x.
          So we can compute P(VarX = x) by adding up P(ω) for all those ω in Omega where VarX(ω) == x.
          total_prob = 0
     '''        
    for omega in Omega:
        if VarX(omega) == x:
            total_prob += P(omega)
    return total_prob
   
    raise NotImplementedError()

def unconditional_joint_probability(Omega, P, EventA):
    ''' P(a) '''
    raise NotImplementedError()

def conditional_probability(Omega, P, VarX, x, VarY, y):
    ''' P(VarX=x|VarY=y) '''
    raise NotImplementedError()

def conditional_joint_probability(Omega, P, EventA, EventB):
    ''' P(a|b) '''
    raise NotImplementedError()

def probability_distribution(Omega, P, VarX, ValX):
    ''' P(VarX),
        which is defined [P(VarX = x0), P(VarX = x1), …] where ValX = [x0, x1, …]
        (return a list)
    '''
    raise NotImplementedError()

def conditional_probability_distribution(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(VarX|VarY)
        to be represented as a python dictionary of the form
        {(x0, y0) : P(VarX=x0|VarY=y0), …}
        for all pairs (x_i, y_j) ∈ ValX × ValY
    '''
    raise NotImplementedError()

def test_event_independence(Omega, P, EventA, EventB):
    ''' P(a,b) = P(a) ⋅ P(b)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    raise NotImplementedError()

def test_variable_independence(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(X,Y) = P(X) ⋅ P(Y)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    raise NotImplementedError()

def test_conditional_independence(Omega, P, EventA, EventB, EventC):
    ''' P(a,b|c) = P(a|c) ⋅ P(b|c)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    raise NotImplementedError()

