class Parameters:
    def __init__(self, beta=11.5, A=20, delta=0.1, nu=1, b=0.01, mu0 = 10, mu1 = 10.45):
        """parameters of the SIR model

        :param beta: incidence, defaults to 11.5
        :type beta: float, optional
        :param A: birth rate, defaults to 20
        :type A: int, optional
        :param delta: natural death rate, defaults to 0.1
        :type delta: float, optional
        :param nu: disease-induced death rate, defaults to 1
        :type nu: int, optional
        :param b: number of beds, defaults to 0.01
        :type b: float, optional
        :param mu0: min recovery rate, defaults to 10
        :type mu0: int, optional
        :param mu1: max recovery rate, defaults to 10.45
        :type mu1: float, optional
        """
        #tolerances
        self.rtol=1e-8
        self.atol=1e-8

        # SIR model parameters
        self.beta=beta
        self.A=A
        self.delta=delta
        self.nu=nu
        self.b=b
        self.mu0 = mu0  # minimum recovery rate
        self.mu1 = mu1  # maximum recovery rate

def B(param):
    """second factor of quadratic equation of endemic equilibrium

    :param param: params of the SIR model
    :type param: Parameters
    :return: float
    :rtype: float
    """
    return (param.delta + param.nu + param.mu0 - param.beta) * param.A + (param.beta - param.nu) * (param.delta + param.nu + param.mu1) * param.b


def A(param):
    """first factor of quadratic equation of endemic equilibrium

    :param param: params of the SIR model
    :type param: Parameters
    :return: float
    :rtype: float
    """
    return (param.delta + param.nu + param.mu0) * (param.beta - param.nu)


def Delta_0(param):
    """discriminant of quadratic equation of endemic equilibrium

    :param param: params of the SIR model
    :type param: Parameters
    :return: float
    :rtype: float
    """
    delta_0 = param.delta + param.nu + param.mu0
    delta_1 = param.delta + param.nu + param.mu1
    return (param.beta - param.nu)**2 * delta_1**2 * (param.b)**2 - 2*param.A*(param.beta - param.nu) * (param.beta*(param.mu1 - param.mu0) + delta_0*(delta_1-param.beta)) * param.b + param.A**2*(param.beta - delta_0)**2


def mu(b, I, mu0, mu1):
    """recovery rate

    :param b: number of bedds
    :type b: float
    :param I: infective people
    :type I: int
    :param mu0: min recovery rate
    :type mu0: float
    :param mu1: max recovery rate
    :type mu1: float
    :return: recovery rate for b and I
    :rtype: float
    """
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu


def R0(beta, delta, nu, mu1):
    """basic reproduction number

    :param beta: incidence
    :type beta: float
    :param delta: death rate
    :type delta: float
    :param nu: disease death rate
    :type nu: float
    :param mu1: max recovery rate
    :type mu1: float
    :return: reproduction number
    :rtype: float
    """
    return beta / (delta + nu + mu1)

def h(I, mu0, mu1, beta, A, delta, nu, b):
    """indicator function for hopf bifurcations

    :param I: infected people
    :type I: float
    :param mu0: min recovery rate
    :type mu0: float
    :param mu1: max recovery rate
    :type mu1: float
    :param beta: incidence
    :type beta: float
    :param A: birht rate
    :type A: float
    :param delta: death rate
    :type delta: float
    :param nu: diesease death rate
    :type nu: float
    :param b: number of beds
    :type b: float
    :return: indicator value
    :rtype: float
    """
    c0 = b**2 * delta * A
    c1 = b * ((mu0-mu1+2*delta) * A + (beta-nu)*b*delta)
    c2 = (mu1-mu0)*b*nu + 2*b*delta*(beta-nu)+delta*A
    c3 = delta*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res


def T(I, param):
    h = h(I, param.mu0, param.mu1, param.beta, param.A, param.delta, param.nu, param.b)
    return - h / ((param.A - param.nu * I) * (I + param.b)**2)
    

def model(t, y, mu0, mu1, beta, A, delta, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    delta
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)
    
    dSdt = A - delta*S - ((beta*S*I) / (S + I + R))
    dIdt = -(delta + nu)*I - m*I + ((beta * S * I) / (S + I + R))
    dRdt = m * I - delta * R
    
    return [dSdt, dIdt, dRdt]