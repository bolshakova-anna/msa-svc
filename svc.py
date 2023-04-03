import cvxpy as cvx
from pandas import DataFrame
from matplotlib import pyplot
import matplotlib.colors as mcolors
import numpy as np

from scipy.integrate import odeint
import scipy.optimize as opt
import time
import quadprog

class SVC():
    def __init__(self,xs,p=0.1,q=1, is_log = False):
        self.is_log = is_log
        self.adj=None
        self.r=None
        self.bsvs=None
        self.svs=None
        self.km=None
        self.beta=None
        self.clusters=None
        self.xs=xs
        self.N=len(xs)
        self.p=p
        self.q=q
        self.C = 1 / (self.p * self.N)
        self.const_sum = None
        self.SEVs = None
        self.indices = None

    # гауссово ядро
    def kernel(self,x1,x2):
        return np.exp(-self.q * np.linalg.norm(x1 - x2)**2 )

    def kernel_matrix(self):
        self.km=np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.km[i,j]=self.kernel(self.xs[i],self.xs[j])

    def get_const_sum(self):
        _ = 0
        for i in range(self.N):
            for j in range(self.N):
                _ +=  self.beta[i]*self.beta[j]*self.km[i,j]
        self.const_sum = _
        # print(self.const_sum)
        
    def find_beta(self):
        beta = cvx.Variable(self.N)
        km = cvx.psd_wrap(self.km)
        objective = cvx.Maximize(cvx.sum(cvx.diag(km) @ beta)- cvx.quad_form(beta, km)) # 1 - cvx.quad_form(beta, km)
        constraints = [0 <= beta, beta<=self.C, cvx.sum(beta)==1]                                   # 1 - self._beta.T @self._km @ self._beta
        result = cvx.Problem(objective, constraints).solve()
        self.beta = np.array(beta.value)
        self.get_const_sum()



    def r_func(self,x):
        return 1 + self.const_sum - 2 * np.sum(
            [
            self.beta[i]*self.kernel(self.xs[i],x) for i in range(self.N)
            ]
            )
    
    def sample_segment(self,x1,x2,n=10):
        for i in range(n):
            x = x1 + (x2-x1)*i/(n+1)
            if self.r_func(x) > (self.r):
                return False
        return True
    
    def init_vectors_and_rad(self):
        svs_tmp = np.array(self.C > self.beta) * np.array(self.beta > 0)
        # print(svs_tmp)
        self.svs=np.where(svs_tmp==True)[0]
        bsvs_tmp=np.array(self.beta==self.C)
        self.bsvs=np.where(bsvs_tmp==True)[0]
       # self.r=np.mean([self.r_func(self.xs[i]) for i in self.svs[:5]])
        self.r = [self.r_func(self.xs[i]) for i in self.svs]
        self.r = np.mean(self.r)

    def cluster(self):
        
        t1 = time.time()
        
        self.adj=np.zeros((self.N,self.N))
        #BSVs не классифицируются этой процедурой, поскольку их изображения лежат вне охватывающей сферы радиуса 
        for i in range(self.N):
            #print(i)
            if i not in self.bsvs:
                for j in range(i,self.N):
                    if j not in self.bsvs:
                        self.adj[i,j]=self.adj[j,i]= self.sample_segment(self.xs[i],self.xs[j])
        
        t = time.time() - t1
        print("Прошло времени:", t)

    def return_clusters(self):
        ids=list(range(self.N))
        self.clusters={}
        num_clusters=-1
        while ids:
            num_clusters+=1
            self.clusters[num_clusters]=[]
            curr_id=ids.pop(0)
            queue=[curr_id]
            while queue:
                cid=queue.pop(0)
                for i in ids:
                    if self.adj[i,cid]:
                        queue.append(i)
                        ids.remove(i)
                self.clusters[num_clusters].append(cid)



    def show_plot(self):
        labels=np.zeros(self.xs.shape[0])
        for i in self.clusters.keys():
            for j in self.clusters[i]:
                labels[j]=int(i)
#        if len(self.clusters)>10:
#            print(f"Number of clusters is more than 10 ({len(self.clusters)})")
#            return 
        df=DataFrame(dict(x=self.xs[:,0],y=self.xs[:,1],label=labels))
        dic=mcolors.CSS4_COLORS
        colors = list(dic.values())
        fig,ax=pyplot.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(15)
        grouped=df.groupby('label')
        for key,group in grouped:
            group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,color=colors[int(key)])
        pyplot.show()


    '''
    LEE
    '''

    def find_SEV(self, x, eps = 1e-6):
        x0 =x 
        res = opt.minimize(self.r_func,x0,method='BFGS', tol=eps)
       # print(f"res = {x}")
        return res.x

    def searchSEV(self,SEV, M, x):
        k = 0 
        for i in range(M+1):
            # если вектора почти совпали по норме, считаем что его уже нашли
            if np.linalg.norm(x-SEV[i]) <= 1e-3:
                k=i
        return k   


    def LeeDecomposingDataIntoGroups(self, grad_err = 1e-5):
        M = -1
        sizeData = np.shape(self.xs)
        rows = 0
        cols = 1
        N = sizeData[rows]
        d = sizeData[cols]
        arr = np.nonzero(self.C > self.beta)  #array of indecces SV and V, not BSV
        indices = -np.ones((sizeData[rows], 1)); #array of indices of SEVs
        SEV = -np.ones((sizeData[rows], sizeData[cols])); #matrix SEV's
        indices[self.beta == self.C] = -2
        t1 = time.time()
        for i in range(N):
            if indices[i] != -2:    
                x0 = self.xs[i]
                x = self.find_SEV(x0,grad_err)
                k = self.searchSEV(SEV,M,x)
                if(k>0):
                    indices[i][0]=k
                else:
                    M+=1
                    indices[i][0]=M
                    SEV[M-1][:] = x
        t = time.time() - t1
        print("Прошло времени:", t)
        #delete useless cells
        if M < sizeData[rows] :
            SEV = SEV[0:M,:]
        self.SEVs = SEV
        self.indices = indices
        #print(f"indices = {indices}")
        
        
    def Part2_Lee(self):
        N = self.N
        N_SEV = len(self.SEVs)
        numCluster = 0
        A2 = np.eye(N_SEV)
        A = np.zeros([N,N])
        #t1
        for i in range(N_SEV):
            min_num = numCluster+1
            for j in range(i-1):
                if A2[i][j] > 0 and A2[i][j] < min_num:
                    min_num = A2[i][j]
            A2[i][i] = min_num
            if min_num > numCluster:
                numCluster = min_num
            for j in range(i+1,N_SEV):
                A2[i][j] = A2[i][j] = self.sample_segment(self.SEVs[i],self.SEVs[j])
        

        for i in range(N):
            if self.beta[i] < self.C:
                for j in range(N):
                    second = self.indices[j]
                    if(second >= 0):
                        A[i][j] = A2[int(self.indices[i]-1)][int(second-1)]
                    else:
                        A[i][j] = 0
                    A[j][i] = A[i][j]
            else:
                A[i][i] = numCluster + 1
            if A[i][i] == 0:
                A[i][i] = numCluster + 1
    

    def my_part_2(self):
        N = self.N
        N_SEV = len(self.SEVs)
        numCluster = 0
        A2 = np.eye(N_SEV)
        A = np.zeros([N,N])
        #t1
        for i in range(N_SEV):
            min_num = numCluster+1
            for j in range(i-1):
                if A2[i][j] > 0 and A2[i][j] < min_num:
                    min_num = A2[i][j]
            A2[i][i] = min_num
            if min_num > numCluster:
                numCluster = min_num
            for j in range(i+1,N_SEV):
                A2[i][j] = A2[i][j] = self.sample_segment(self.SEVs[i],self.SEVs[j])
        self.adj = A2
        
        ids=list(range(N_SEV))
        self.clusters={}
        num_clusters=-1
        while ids:
            num_clusters+=1
            self.clusters[num_clusters]=[]
            curr_id=ids.pop(0)
            queue=[curr_id]
            while queue:
                cid=queue.pop(0)
                for i in ids:
                    if self.adj[i,cid]:
                        queue.append(i)
                        ids.remove(i)
                self.clusters[num_clusters].append(cid)
        
       
#        if len(self.clusters)>10:
#            print(f"Number of clusters is more than 10 ({len(self.clusters)})")
#            return 

    def show_Lee(self):
        labels=np.zeros(self.xs.shape[0])
        for i in self.clusters.keys():
            for j in self.clusters[i]:
                labels[np.where(self.indices == j)[0]]=int(i)
        df=DataFrame(dict(x=self.xs[:,0],y=self.xs[:,1],label=labels))
        dic=mcolors.TABLEAU_COLORS
        colors = list(dic.values())
        fig,ax=pyplot.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(15)
        grouped=df.groupby('label')
        for key,group in grouped:
            group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,marker = 'o',color=colors[int(key)])
        ax.scatter(self.SEVs[:, 0], self.SEVs[:, 1], c='r', marker='d', label="SEV")
        pyplot.show()
