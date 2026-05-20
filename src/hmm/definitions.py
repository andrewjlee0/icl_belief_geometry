"""HMM family definitions: main, order-1, and order-0 transition matrices."""
import numpy as np

# ── Mess3: 3 states, 3 tokens ──
def mess3_matrices(alpha, x):
    b = (1 - alpha) / 2; y = 1 - 2 * x
    ay, bx, by, ax = alpha*y, b*x, b*y, alpha*x
    return [
        np.array([[ay,bx,bx],[ax,by,bx],[ax,bx,by]]),
        np.array([[by,ax,bx],[bx,ay,bx],[bx,ax,by]]),
        np.array([[by,bx,ax],[bx,by,ax],[bx,bx,ay]]),
    ]
def mess3_order_one(alpha, x):
    A = 0.5*(1-2*alpha+3*alpha**2-x+6*alpha*x-9*alpha**2*x)
    B = 0.25*(1+2*alpha-3*alpha**2+x-6*alpha*x+9*alpha**2*x)
    return [np.array([[A,0,0],[B,0,0],[B,0,0]]),np.array([[0,B,0],[0,A,0],[0,B,0]]),np.array([[0,0,B],[0,0,B],[0,0,A]])]

# ── Arch: 4 states, 3 tokens ──
def arch_matrices(alpha):
    b = (1 - alpha) / 3
    return [
        np.array([[0.8*alpha,0,0,0],[0,0.2*alpha,0,0],[0,0,0.4*alpha,0],[0,0,0,0.6*alpha]]),
        np.array([[0,0,0,0],[0,0.4*alpha,0,0.4*b],[0,0,0.3*alpha,0],[0,0,0,0.16*alpha]]),
        np.array([[0.2*alpha,b,b,b],[b,0.4*alpha,b,0.6*b],[b,b,0.3*alpha,b],[b,b,b,0.24*alpha]]),
    ]
def arch_order_one(alpha):
    d1=20+109*alpha; d2=-580+409*alpha
    return [
        np.array([[3*alpha/5,0,0],[6*alpha*(10+27*alpha)/(5*d1),0,0],[18*alpha*(-80+59*alpha)/(5*d2),0,0]]),
        np.array([[0,(10+101*alpha)/750,0],[0,alpha*(560+1507*alpha)/(50*d1),0],[0,(-1000-4690*alpha+3527*alpha**2)/(50*d2),0]]),
        np.array([[0,0,(740-551*alpha)/750],[0,0,(1000+4290*alpha-3127*alpha**2)/(50*d1)],[0,0,-(28000-39540*alpha+14147*alpha**2)/(50*d2)]]),
    ]
def arch_order_zero(alpha):
    p1=alpha/2; p2=(20+109*alpha)/600
    return [np.array([[p1]]),np.array([[p2]]),np.array([[1-p1-p2]])]

# ── Wing: 3 states, 2 tokens ──
def wing_matrices(alpha, x):
    b = (1 - alpha) / 2
    return [
        np.array([[0,b,0],[0,x*alpha,0.5*b],[b,0,0]]),
        np.array([[alpha,0,b],[b,(1-x)*alpha,0.5*b],[0,b,alpha]]),
    ]
def wing_order_one(alpha, x):
    p=2-4*alpha+2*alpha**2+3*alpha*x-3*alpha**2*x+4*alpha**2*x**2
    q=-3+alpha+2*alpha**2-alpha*x-3*alpha**2*x+4*alpha**2*x**2
    r=4+6*alpha+2*alpha**2-5*alpha*x-3*alpha**2*x+4*alpha**2*x**2
    d1=5-5*alpha+4*alpha*x; d2=-7-5*alpha+4*alpha*x
    return [np.array([[p/d1,0],[q/d2,0]]),np.array([[0,-q/d1],[0,-r/d2]])]
def wing_order_zero(alpha, x):
    d1=5-5*alpha+4*alpha*x; d2=7+5*alpha-4*alpha*x
    return [np.array([[d1/12]]),np.array([[d2/12]])]

# ── Strata: 3 states, 2 tokens ──
def strata_matrices(alpha, t0, t1):
    b = (1 - alpha) / 2
    return [
        np.array([[t0*alpha,0,0],[0,t1*alpha,0],[0,0,0]]),
        np.array([[(1-t0)*alpha,b,b],[b,(1-t1)*alpha,b],[b,b,alpha]]),
    ]
def strata_order_one(alpha, t0, t1):
    n1=alpha*(t0**2+t1**2); n2=-t0+alpha*t0**2-t1+alpha*t1**2
    n3=3-2*alpha*t0+alpha**2*t0**2-2*alpha*t1+alpha**2*t1**2
    d1=t0+t1; d2=-3+alpha*t0+alpha*t1
    return [np.array([[n1/d1,0],[alpha*n2/d2,0]]),np.array([[0,-n2/d1],[0,-n3/d2]])]
def strata_order_zero(alpha, t0, t1):
    p=alpha*(t0+t1)/3
    return [np.array([[p]]),np.array([[1-p]])]

# ── Spiral: 3 states, 2 tokens ──
def spiral_matrices(alpha):
    return [
        np.array([[0.2*alpha,0,0],[0,0,0],[0.25*(1-alpha),0,0.5*alpha]]),
        np.array([[0.8*alpha,1-alpha,0],[0,alpha,1-alpha],[0.75*(1-alpha),0,0.5*alpha]]),
    ]
def spiral_order_one(alpha):
    n1=alpha*(35+23*alpha); n2=-50-55*alpha+23*alpha**2; n3=500-145*alpha+23*alpha**2
    d1=10*(5+9*alpha); d2=10*(-55+9*alpha)
    return [np.array([[n1/d1,0],[n2/d2,0]]),np.array([[0,-n2/d1],[0,-n3/d2]])]
def spiral_order_zero(alpha):
    return [np.array([[(5+9*alpha)/60]]),np.array([[(55-9*alpha)/60]])]
