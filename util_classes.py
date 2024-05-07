import numpy as np
import matplotlib.pyplot as plt

'''util function'''
def random_pick(choices, probs):
  # draws a random choice according to probs
  cutoffs = np.cumsum(probs)
  idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))
  return choices[idx]

# 1. Hospital class the OR class

class Hospital:

    def __init__(self, N=10):
        assert(isinstance(N,int))
        self.N = N
        self.policy = 'Flexible'
        self.ORs = [0]*N
        for i in range(N):
            self.ORs[i] = OR(i, self.policy)
        self.q = []
        self.backlog = []
    
    def sortBg(self):
        if len(self.backlog) > 0:
          self.backlog  = sorted(self.backlog, key=lambda p: p.arrive)

    def tryPatient(self,p,time):
        # assert(isinstance(p,Patient))
        for i in range(self.N):
          res = self.ORs[i].tryPatient(p,time)
          if res: 
            return [True,i]
        return [False,-1]
 
    def allFull(self):
        res = True; num = 0
        for OR in self.ORs:
            if OR.isfull is False:
                res = False
            else: #this one is full
                num += 1
        if num < self.N: assert(res is False)
        return [res,num]

    def checkORstate(self):
        out = []
        for i in range(self.N):
            x = self.ORs[i]
            if x.isfull:
                res = [f'ORind:{i}', f'isfull:{x.isfull}',
                   f'p cls:{x.p.cls}', f'p remainTime:{x.p.ORtime - x.p.ORtick}']
                if x.p.ORtime - x.p.ORtick == 0:
                    print(x.p.ORind, x.ind)
                    print(res)
                    x.releasePatient(x.p)
            else:
                res = [f'ORind:{i}', f'isfull:{OR.isfull}']
            out.append(res)
        print('\n________')
        for text in out:
            print('-----')
            print(text)
            print('-----')
        print('________\n')


class OR:
    policies = {'NonE','Elec','Flexible','Priority'}
    def __init__(self, ind = -1, policy=None):
        self.policy = policy
        self.schedule = None # <--- needs implementation
        self.isfull = False
        self.ind = ind

    def setP(self, new_p=None):
        if new_p is not None:
            self.policy = new_p

    def setBreak(self,breaks=None):
        if breaks is None:
          self.breaks = [(288,360),(648,720),(1008,1080),(1368,1440)]
        else:
          assert(isinstance(breaks,list))
          assert(len(breaks)<=4)
          self.breaks = breaks

    def tryPatient(self, p, time):
        # assert(isinstance(p,Patient))
        if self.isfull or p.inOR: return False
        if p.cls == 'I': return True

        if self.policy in ['NonE','Elec']:
          return p.stat == self.policy
        elif self.policy == 'Priority':
            for br in self.breaks:
                if br[0] <= time <= br[1]: #within break, priority=NonE
                    return p.stat == 'NonE'
            #not in break, priority=Elec
            return p.stat == 'Elec'
        else: #flexible
            return True
        
    def takePatient(self,p):
        # assert(isinstance(p,Patient))
        assert(self.isfull is False)
        # print(f'##### in takePatient:{p}')
        p.inOR = True
        p.ORind = self.ind
        self.isfull = True
        self.p = p
        return p

    def releasePatient(self,p):
        # assert(isinstance(p,Patient))
        assert(self.isfull)
        assert((p.ORind == self.ind) and p.ORtick >= p.ORtime)
        p.inOR = False
        if p.cls == 'I':
            p.cls = 'R' #recovered
        p.finish = True
        self.p = None
        self.isfull = False
        return p
        
          

# 2. patient class

class Patient:

  # 'Elec' = elective = add-on in Antognini et al. (2015)
  # 'NonE' = non-elective = emergent in Antognini et al. (2015)
  

  def __init__(self, cls = 'S',limit_day = 1):
    self.sgTime = {'Elec':[5.01655,0.713405],'NonE':[5.00716, 0.583642]}
    self.cls = cls
    self.inOR = False; self.ORind = -1
    self.stat = random_pick(['Elec','NonE'], [0.8,0.2])
    assert(self.stat in self.sgTime.keys())
    if self.cls == 'I':
      self.stat = 'NonE'
      self.limit = 720 #must do surgery in 12hr
    elif self.stat == 'NonE':
      self.limit = 1440 #24hr
    else:
      self.limit = 1440*limit_day #24*60min * 5
    self.tick = 0; self.ORtick = 0; self.arrive = -1
    self.hasArrived = False; self.finish = False

  def getTime(self):
    if self.cls == 'S':
      self.ORtime = int(np.random.lognormal(self.sgTime[self.stat][0], 
                                            self.sgTime[self.stat][1]))
    elif self.cls == 'I':
      self.ORtime = int(np.random.lognormal(self.sgTime['NonE'][0], 
                                            self.sgTime['NonE'][1]))

  def tickTock(self):
    if self.hasArrived:
        if (self.inOR is False) and (self.finish is False):
            self.tick += 1
        if (self.inOR is False) and (self.tick >= self.limit):
        #gg
            self.precls = self.cls
            self.cls = 'D'
    if self.inOR:
        self.ORtick += 1
        # print('???')


# 3. SIR model class

class SIRModel:

    def __init__(self, N=1e4, init = 1, alpha=0.1, beta=0.0005, d0 = 0,
                T=70, dt=1):
        assert(isinstance(N,int))
        assert(0 < alpha < 1)
        assert(0 < beta < 1)
        assert(0 < dt < T)
        self.N = N; self.alpha = alpha
        self.beta = beta; self.d0 = d0
        self.T = T; self.dt = dt
        self.len = int(T/dt)+1
        self.contSeries = {'S':[self.N-init], 'I':[init], 'R':[0], 'D':[0]}
        self.timeSeries = {'S':[self.N-init], 'I':[init], 'R':[0], 'D':[0]}
        self.incremI = [[1],[1]] #(int, float) version of dI

    def checkLength(self):

        res = len(self.timeSeries['S']) == len(self.timeSeries['I']) == len(self.timeSeries['R']) == len(self.timeSeries['D'])
        return (res and len(self.timeSeries['S']) < self.len)

    def getLengths(self):

        return [len(self.timeSeries['S']), len(self.timeSeries['I']),
                len(self.timeSeries['R']), len(self.timeSeries['D'])]

    def oneStep(self, S_cur = None, I_cur = None, R_cur = None, D_new = None):
        #use forward Euler discreization to approximate ODE
        '''f(t+dt)-f(t) = df(t)*dt'''
        #just adding 1 time step

        if self.checkLength():
            if S_cur is None:
                S_cur = self.contSeries['S'][-1]
            if I_cur is None:
                I_cur = self.contSeries['I'][-1]
            if R_cur is None:
                R_cur = self.contSeries['R'][-1]

            dS = (-self.beta * S_cur * I_cur) * self.dt #dS(t-1) = -bSI(t-1)
            dI = (self.beta * S_cur * I_cur - (self.alpha+self.d0) * I_cur) * self.dt #dI = bSI - aI
            dR = (self.alpha * I_cur) * self.dt # dR = aI
            # print(S_cur, dS)
            if S_cur+dS <=0 : 
                S_new = 0; I_new = I_cur + S_cur; R_new = R_cur + dR
            else:
                S_new = S_cur + dS; I_new = I_cur + dI; R_new = R_cur +dR
            
            if D_new is None:
                dD = self.d0 * I_cur
                D_new = self.contSeries['D'][-1]+dD
            self.incremI[0].append(int(dI)); self.incremI[1].append(dI)
            #round up integer version
            self.timeSeries['S'].append(max(int(S_new),0))
            self.timeSeries['I'].append(max(int(I_new),0))
            self.timeSeries['R'].append(max(int(R_new),0))
            self.timeSeries['D'].append(max(int(D_new),0))
            #float version
            self.contSeries['S'].append(S_new)
            self.contSeries['I'].append(I_new)
            self.contSeries['R'].append(R_new)
            self.contSeries['D'].append(D_new)
            #correction for rounding errors
            diff = self.N - int(S_new) - int(I_new) - int(R_new) - int(D_new)
            self.timeSeries['D'][-1] += diff
            assert(int(S_new)+int(I_new)+int(R_new)+int(D_new)+int(diff)==self.N)
            self.contSeries['S'][-1] += int(diff)
        else:
            print('Length check failed')
            print(f'Max. length: {self.len}')
            print(f'SIRD Lengths:{self.getLengths()}')

    def last_dI(self):
        return max(0,self.incremI[0][-1])
    
    def lastOut(self):
        return [self.timeSeries['S'][-1], self.timeSeries['I'][-1],
                self.timeSeries['R'][-1], self.timeSeries['D'][-1]]
    
def roundhalf(x):
    '''round up if decimal part >= 0.5, round down o.w.'''
    n = np.ceil(x)
    if n-x > 0.5: return int(np.floor(x))
    else: return int(n)

def plot_2D(data_info,title,input_label,output_label,
            axis_bounds=None,xscale=None,yscale=None,bloc='best',save=(False,'')):
    '''
    NOTES: Plots multiple 2D data on one graph.
    INPUT: 
        data_info = list of lists with structure:
            ith list = ith data information, as list
            ith list[0] = [input, output]
            ith list[1] = desired color for ith data
            ith list[2] = legend label for ith data
        title = string with desired title name
        input_label = string with name of input data
        output_label = string with name of output data
        axis_bounds = list with structure: [xmin, xmax, ymin, ymax]
        xscale = string with x axis scale description
        yscale = string with y axis scale description
    '''
    fig = plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    for info_cache in data_info:
        if len(data_info) > 1:
            alp = 0.8
            mksize = 10
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = alp)
        else:
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=10)

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    if axis_bounds is not None:
        plt.axis(axis_bounds)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend(loc=bloc,fontsize=14)
    plt.show()
    if save[0]:
        fig.savefig(save[1]+'_.png')

def plot_hist(data_info,title,input_label,output_label,axis_bounds=None,xscale=None,yscale=None):
    '''
    NOTES: Plots one histogram. (and the average line)
    '''
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    plt.hist(data_info[0], color='c', edgecolor='k', alpha=0.65)
    plt.axvline(data_info[1], color='k', linestyle='dashed', linewidth=2)

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    if not axis_bounds == None:
        plt.axis(axis_bounds)
    if not xscale == None:
        plt.xscale(xscale)
    if not yscale == None:
        plt.yscale(yscale)
    plt.legend(loc='best',fontsize=14)
    plt.show()

def plot_2Dy2(data_info,data_info2,title,input_label,output_label,output_label2,
              axis_bounds=None,xscale=None,yscale=None):
    '''
    NOTES: Plots multiple 2D data on one graph.
    INPUT: 
        data_info = list of lists with structure:
            ith list = ith data information, as list
            ith list[0] = [input, output]
            ith list[1] = desired color for ith data
            ith list[2] = legend label for ith data
        data_info2 = data for second plot
        title = string with desired title name
        input_label = string with name of input data
        output_label = string with name of output data
        output_label2 = string with name of second output data
        axis_bounds = list with structure: [xmin, xmax, ymin, ymax]
        xscale = string with x axis scale description
        yscale = string with y axis scale description
    '''
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    style1 = []
    label1 = []
    mksize = max(int(20/len(data_info[0][0][0])),5)
    for info_cache in data_info:
        
        
        plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = 0.8)
        style1.append(info_cache[1])
        label1.append(info_cache[2])
        dummy = [info_cache[0][0][0],info_cache[0][1][0]]

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    
    plt2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
    
    plt2.set_ylabel(output_label2,fontsize=20)  # set secondary y axis
    for i in range(len(data_info)):
        data_info2.append([dummy,style1[i],label1[i]])
    mksize = max(int(20/len(data_info2[0][0][0])), 5)
    for info_cache in data_info2:
         
        plt2.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = 0.65)
    plt2.tick_params(axis='y')
    
    if axis_bounds is not None:
        plt.axis(axis_bounds)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend(loc='best',fontsize=14)
    plt.show()