import numpy as np
from scipy import stats as st

def consecutiveRanges(a):
    n=len(a)
    length = 1;list = []
    if (n == 0):
        return list
    for i in range (1, n + 1):
        if (i == n or a[i] - a[i - 1] != 1):
            if (length > 0):
                if (a[i - length]!=0):
                    temp = (a[i - length]-1, a[i - 1])
                    list.append(temp)
            length = 1
        else:
            length += 1
    return list
def Extract(lst):
    return list(list(zip(*lst))[0])
def Extract_end(lst):
    return list(list(zip(*lst))[1])


def go_avalanches(data,thre=3.,direc=0,binsize=1):
    if direc==1:
        Zb=np.where(st.zscore(data)>thre,1,0)
    elif direc==-1:
        Zb=np.where(st.zscore(data)<-thre,1,0)
    elif direc==0:
        Zb=np.where(np.abs(st.zscore(data))>thre,1,0)
    else:
        print('wrong direc')
    nregions=len(data[0])
    Zbin=np.reshape(Zb,(-1,binsize,nregions))
    Zbin=np.where(np.sum(Zbin,axis=1)>0,1,0)
    dfb_ampl=np.sum(Zbin,axis=1).astype(float)
    dfb_a=dfb_ampl[dfb_ampl!=0]
    bratio=np.exp(np.mean(np.log(dfb_a[1:]/dfb_a[:-1])))
    NoAval=np.where(dfb_ampl==0)[0]
    inter=np.arange(1,len(Zbin)+1); inter[NoAval]=0
    Avals_ranges=consecutiveRanges(inter)
    Avals_ranges=Avals_ranges[1:-1] #remove the first and last avalanche for avoiding boundary effects
    Naval=len(Avals_ranges)   #number of avalanches
    Avalanches={'dur':[],'siz':[],'ranges':Avals_ranges,'Zbin':Zbin,'bratio':bratio} #duration and size
    for i in range(Naval):
        xi=Avals_ranges[i][0];xf=Avals_ranges[i][1]
        Avalanches['dur'].append(xf-xi)
        Avalanches['siz'].append(np.sum(Zbin[xi:xf]))
    return Avalanches