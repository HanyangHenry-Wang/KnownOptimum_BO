# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:05:06 2020

@author: Vu Nguyen
"""

from random import sample
import numpy as np
from scipy.stats import norm

class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq_name):
                
        ListAcq=['bucb','ucb', 'ei', 'poi','random','thompson', 'lcb', 'mu',                     
                     'pure_exploration','kov_mes','mes','kov_ei','gp_ucb',
                         'erm','cbm','kov_tgp','kov_tgp_ei','find0','truncated_mean_ei','findfmax','truncated_ei','MC_ei']
        # kov = know optimum value
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
                    
    def acq_kind(self,gp,x):
            
        y_max=np.max(gp.Y)
        
        if np.any(np.isnan(x)):
            return 0
        
        if self.acq_name == 'ucb' or self.acq_name == 'gp_ucb' :
            return self._gp_ucb( gp,x)
        if self.acq_name=='cbm':
            return self. _cbm(x,gp,target=gp.fstar)
        if self.acq_name == 'lcb':
            return self._lcb(gp,x)
        if self.acq_name == 'ei' or self.acq_name=='kov_tgp_ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'kov_ei' :
            return self._ei(x, gp, y_max=gp.fstar)
        if self.acq_name == 'erm'  or self.acq_name=='kov_ei_cb':
            return self._erm(x, gp, fstar=gp.fstar)
        
        if self.acq_name == 'find0' : #我自己定义的
            return self._find0 (x, gp)
        
        if self.acq_name == 'truncated_mean_ei' : #我自己定义的
            return self._truncated_mean_ei (x, gp,y_max, fstar=gp.fstar)
        
        if self.acq_name == 'truncated_ei' : #我自己定义的
            return self._truncated_ei (x, gp,y_max, fstar=gp.fstar)
        
        if self.acq_name == 'findfmax' : #我自己定义的
            return self._findfmax (x, gp, fstar=gp.fstar)
        
        if self.acq_name == 'MC_ei' : #我自己定义的
            return self._MC_ei (x, gp, y_max, fstar=gp.fstar)
    
    @staticmethod
    def _lcb(gp,xTest,fstar_scale=0):
        mean, var = gp.predict(xTest)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0

        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));
        
        output = mean - np.sqrt(beta_t) * np.sqrt(var) 
    
        return output.ravel()
    
    @staticmethod
    def _gp_ucb(gp,xTest,fstar_scale=0):
        #dim=gp.dim
        #xTest=np.reshape(xTest,(-1,dim))
        mean, var = gp.predict(xTest)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10] = 0             
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
      
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        output = mean + np.sqrt(beta_t) * np.sqrt(var)
        #print("input",xTest.shape,"output",temp.shape)
        return  output.ravel()
    
    @staticmethod
    def _cbm(x, gp, target): # confidence bound minimization
        mean, var = gp.predict(x)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0            
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
      
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        output = -np.abs(mean-target) - np.sqrt(beta_t) * np.sqrt(var) 
        return output.ravel()
       
    @staticmethod
    def _erm(x, gp, fstar):
                
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
    
        var2 = np.maximum(var, 1e-9 + 0 * var)
        z = ( fstar-mean)/np.sqrt(var2)        
        out=(fstar-mean) * (norm.cdf(z)) + np.sqrt(var2) * norm.pdf(z)

        return -1*out.ravel() # for minimization problem
                    
    @staticmethod
    def _ei(x, gp, y_max):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        return out.ravel()
    
    
    @staticmethod
    def _find0(x, gp):  #我自己定义的
        #y_max=np.asscalar(y_max)
        meanG, varG = gp.predict_G(x)
        pdf_0 = 1/(np.sqrt(2*np.pi*varG))*np.exp(-meanG**2/(2*varG))
        
        #print(out.shape)
        return pdf_0.ravel()  
    
    
    @staticmethod
    def _truncated_mean_ei(x, gp, y_max, fstar):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        
        truncated_mean = np.minimum(mean,fstar)
        
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (truncated_mean - y_max)/np.sqrt(var2)        
        out=(truncated_mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        return out.ravel()
    
    @staticmethod
    def _truncated_ei(x, gp, y_max, fstar): ##############
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        
        z1 = (mean - y_max)/np.sqrt(var2)  
        z2 = (fstar - y_max)/np.sqrt(var2)  
        
        out1=(mean - y_max) * norm.cdf(z1) + np.sqrt(var2) * norm.pdf(z1)
        out2 = (fstar - y_max) * norm.cdf(z2) + np.sqrt(var2) * norm.pdf(z2)
        
        out = out1-out2
        out[out < 0] = 0
        
       # out[var2<1e-10]=0
        
        return out.ravel()
    
    
    
    @staticmethod
    def _findfmax(x, gp, fstar):  #我自己定义的
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        pdf_fmax = 1/(np.sqrt(2*np.pi*var))*np.exp(-(fstar-mean)**2/(2*var))
        
        #print(out.shape)
        return pdf_fmax.ravel()
    
    @staticmethod
    def _MC_ei(x, gp, y_max, fstar):  #我自己定义的
        #y_max=np.asscalar(y_max)
        meanG, varG = gp.predict_G(x)
        covG = np.diag(varG)
        sampleG = np.random.multivariate_normal(meanG, covG)
        samplef = fstar-0.5*sampleG**2
    
        
        sum = np.zeros(20)
        for i in range(20):
            sampleG = np.random.multivariate_normal(meanG, covG)
            samplef = fstar-0.5*sampleG**2
            sample_ei = samplef-y_max
            sample_ei[sample_ei < 0] = 0
            
            sum = sum+sample_ei
            
        res = sum/20
                
        return res.ravel()  
