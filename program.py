'''Aux. util functions'''

def estimateN(days=70,numOR=10,util = 0.8,r=0.15):
    #assumed utilization, and # surgery/population = r
    return int(((numOR*util*24*days)/(np.exp(5.1)/60))/r)

def getPatients(n=10,cls='S',prob=0.15,days=70,cur_day=0,limit=5):
    #good to go
    assert(isinstance(n,int))
    assert(0 <= prob <= 1)
    res = []
    if n == 0: return res
    for _ in range(n):
        if cls == 'S':
            r = np.random.uniform(0,1)
            p = Patient(cls,limit_day=5) if r <= prob else 0 #
            if isinstance(p,Patient):
                p.getTime() #generate surgery time
                p.arrive = np.random.randint(days)*24*60+np.random.randint(24*60) #arrival time in whole interval
                res.append(p)
        elif cls =='I':
            p = Patient(cls)
            p.getTime()
            p.arrive =  np.random.randint(24*60)+cur_day*24*60 #arrival time (fix date)
            res.append(p)
    res_sorted = sorted(res, key=lambda p: p.arrive) #in ascending order of arrival time
    return res_sorted

def genSurgery(numOR=10,util=0.8,limit=5):
    #generate random patients
    sgMinute = numOR * util * 24 * 60
    patients = []
    total = 0
    while total < sgMinute:
        new_p = Patient(cls='S',limit_day = limit)
        new_p.stat = 'Elec'
        new_p.getTime()
        new_p.arrive = 0
        total += new_p.ORtime
        patients.append(new_p)
    return patients




#Simulation main program:
def program(days = 70, numOR = 10, init = 1, alpha=0.05, beta=0.00005, d0 = 0, sg_prob=0.3,
            util = 0.8, limit = 5,
           debug = True):
    
    N = estimateN(days,numOR,util)
    print(f'Estimated population :{N}\n')
    SIR = SIRModel(N, init, alpha, beta, d0, days)
    hospital = Hospital(numOR) #defaul policy = flexible
    S = SIR.timeSeries['S'][-1]
    #add all S-patients into hospital queue
    hospital.backlog += getPatients(S,cls='S',prob=sg_prob,days=days)
    print(f'Backlog S patients # :{len(hospital.backlog)}, percent = {(len(hospital.backlog)/S) * 100}\n')
    
    hospital.q += genSurgery(numOR,0.5,limit)
    
    
    #timeseries data and waittime data
    q_time = []
    bg_time = []
    fullnum_time = []
    WT_time = {'NonE':[],'Elec':[],'I':[]}
    
    # time loop begins
    minute_ct = 0
    day_ct = 0
    checkQ = False
    
    for d in range(days):
        
        I_today = SIR.timeSeries['I'][-1]
        dI = SIR.last_dI()
        S_today = SIR.timeSeries['S'][-1]
        D_today = SIR.timeSeries['D'][-1]
        R_today = SIR.timeSeries['R'][-1]
        if debug: 
            print(f'BEGIN day {d} -- we have I:{I_today}, S:{S_today}, R:{R_today}, D:{D_today}\n')
#             print(f'Hospital queue size: {len(hospital.q)}\n')
        #add all I-patients of the day
        hospital.backlog += getPatients(n=dI,cls='I',cur_day=d)
        hospital.sortBg()

        for m in range(24*60): #minutes in a day
            
            while len(hospital.backlog) > 0:
                patient = hospital.backlog[0]
                if patient.arrive <= minute_ct: #check if patient has arrived
                    patient.hasArrived = True
                    if patient.cls == 'I':
                        checkQ = True
                    hospital.q.append(patient)
                    hospital.backlog.pop(0)
                else: break #no need to check further, hasn't arrived yet
            
            if checkQ: 
                hospital.q = sorted(hospital.q, key=lambda x: (x.cls!='I',x.arrive))
                assert(hospital.q[0].cls == 'I')
                checkQ = False
  
            poplist = []
            [allFull, num] = hospital.allFull()

            if allFull is False: #try taking patient to OR
                for i in range(len(hospital.q)):
                    patient = hospital.q[i]
                    [canTake,ind] = hospital.tryPatient(patient,m)
                    if canTake: 
                        patient = hospital.ORs[ind].takePatient(patient)
                        assert(patient.ORind == ind)
                        hospital.q[i] = patient
                        assert(patient.inOR)
                        
                        
                        if patient.cls == 'S':
                            WT_time[patient.stat].append(patient.tick)
                        elif patient.cls == 'I':
                            WT_time['I'].append(patient.tick)
        
#             if day_ct > 1 and R_today == SIR.timeSeries['R'][-2]: hospital.checkORstate()
            
            for i in range(len(hospital.q)):
                patient = hospital.q[i]
                if patient.cls == 'D': #check if patient has gg
                    assert(patient.tick >= patient.limit)
                    D_today += 1

                    if patient.precls == 'S' and S_today >= 1:
                        S_today -= 1
                        WT_time[patient.stat].append(patient.limit)
                    elif patient.precls == 'I' and I_today >= 1:
                        I_today -= 1
                        WT_time['I'].append(patient.limit)
                    
                    poplist.append(i) #gg byebye

                elif patient.ORtick >= patient.ORtime: 
                    patient = hospital.ORs[patient.ORind].releasePatient(patient)
                    hospital.q[i] = patient
                    assert(patient.finish)
                    if patient.cls == 'R':
                        R_today += 1
                        I_today -= 1
                    poplist.append(i) #finished byebye
                    
            for ind in sorted(poplist, reverse=True):
                hospital.q.pop(ind)

            for i in range(len(hospital.q)):
                hospital.q[i].tickTock() #for patients that has arrived

            minute_ct += 1    
            if allFull: fullnum_time.append(numOR)
            else: fullnum_time.append(num)
            q_time.append(len(hospital.q))
            bg_time.append(len(hospital.backlog))
            
        day_ct += 1    
        if debug: print(f'END day {d} -- we have I:{I_today}, S:{S_today}, R:{R_today}, D:{D_today}\n')
        assert(S_today+I_today+R_today+D_today == N)
        SIR.oneStep(S_today,I_today,R_today,D_today)
#         if day_ct > 8: break
        
    return [day_ct, minute_ct], SIR, fullnum_time, q_time, bg_time, WT_time
