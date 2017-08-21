import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math


def get_long_mom_nu(MET,proc,event_id):
    MW=80.301
    
    px_Nu=MET[proc]['pT'][event_id]*math.cos(MET[proc]['Phi'][event_id])
    py_Nu=MET[proc]['pT'][event_id]*math.sin(MET[proc]['Phi'][event_id])

    met=MET['sig']['pT'][event_id]

    px_lep=Lepton[proc]['pT'][event_id]*math.cos(Lepton[proc]['Phi'][event_id])
    py_lep=Lepton[proc]['pT'][event_id]*math.sin(Lepton[proc]['Phi'][event_id])
    pz_lep=Lepton[proc]['pT'][event_id]*math.sinh(Lepton[proc]['Eta'][event_id])
    E_lep =Lepton[proc]['pT'][event_id]*math.cosh(Lepton[proc]['Eta'][event_id])

    P_lep=math.sqrt(px_lep**2 + py_lep**2 + pz_lep**2)
    P_lep_T=Lepton[proc]['pT'][event_id]

    AW=MW**2 + 2.0*(px_lep*px_Nu + py_lep*py_Nu)
    Discrim=AW**2 - 4.0*(P_lep_T*met)**2

    pNu_p=[]
    pNu_m=[]
    if Discrim<0:
        Discrim = -Discrim
        
    pz_Nu_p=(0.5/(P_lep_T**2))*(AW*pz_lep + E_lep*math.sqrt(Discrim))
    pz_Nu_m=(0.5/(P_lep_T**2))*(AW*pz_lep - E_lep*math.sqrt(Discrim))

    E_Nu_p=math.sqrt(pz_Nu_p**2 + met**2)
    E_Nu_m=math.sqrt(pz_Nu_m**2 + met**2)

    pNu_p.extend([px_Nu,py_Nu,pz_Nu_p,E_Nu_p])
    pNu_m.extend([px_Nu,py_Nu,pz_Nu_m,E_Nu_m])
    
    return pNu_p,pNu_m

#Function 'get_invariant_mass()' takes any number of particle momenta and 
#returns the invariant mass of the group of particles
def get_invariant_mass(*args):
    tmp_x=0.0
    tmp_y=0.0
    tmp_z=0.0
    tmp_E=0.0
    
    for i,e in enumerate(args):
        tmp_x= tmp_x + float(e[0])
        tmp_y= tmp_y + float(e[1])
        tmp_z= tmp_z + float(e[2])
        tmp_E= tmp_E + float(e[3])

    if(tmp_E**2-tmp_x**2-tmp_y**2-tmp_z**2)>0.0:
        Inv_Mass=math.sqrt(tmp_E**2-tmp_x**2-tmp_y**2-tmp_z**2)
    else:
        Inv_Mass=0.0
    return Inv_Mass
#----------------------------------------------------
def vars_event(part_mom):
    part_E   = [e[0]*math.cosh(e[1]) for e in part_mom]
    part_pT  = [e[0] for e in part_mom]
    part_eta = [e[1] for e in part_mom]
    return part_E,part_pT,part_eta
#----------------------------------------------------
def to_cartesion(*args):
    p_=[]
    for p in args:
        px=p['pT']*math.cos(p['Phi'])
        py=p['pT']*math.sin(p['Phi'])
        pz=p['pT']*math.sinh(p['Eta'])
        E =p['pT']*math.cosh(p['Eta'])
    
    p_.extend([px,py,pz,E])
    return p_
#----------------------------------------------------
def jet_vars(dta,jet_id):
    
    Jet=pd.Series([dta['Jet'][i][jet_id] for i in range(len(dta))]).apply(pd.Series)
    Jet.columns=['pT','Eta','Phi','BTag']
    Jet['E']=Jet['pT']*Jet['Eta'].map(math.cosh)
    Jet.reindex(columns=['pT','Eta','Phi','E','PId'])
    return Jet
#----------------------------------------------------
def remove_nan_events(dta):
##Function to remove nan events from the data file
    events_to_remove=[]
    for i in range(len(dta)):
        if type(dta['Jet'][i]) is not list:
            events_to_remove.append(i)

    
    dta=dta.drop(dta.index[events_to_remove])
    dta.reset_index(drop=True,inplace=True)
    return dta,events_to_remove

def Var_plot(part_mom, figsize,lbl1):
    fig = plt.figure(figsize=figsize)
    mpl.rcParams['font.family'] = 'Ubuntu Mono'
    layout = (1, 3)
    
    E_ax   = plt.subplot2grid(layout, (0, 0))
    E_ax.set_xlabel('Energy (GeV)')
    
    pT_ax  = plt.subplot2grid(layout, (0, 1))
    pT_ax.set_xlabel('Tranverse Momentum (GeV)')
    
    eta_ax = plt.subplot2grid(layout, (0, 2))
    eta_ax.set_xlabel('Rapidity')

    part_E, part_pT, part_eta = vars_event(part_mom)       

    hist_params={'histtype':'step',
                 'bins':50,
                 'normed':True,
                 'label':'Signal'}
    
    E_ax.hist(part_E, 
              range=[10,1000],
              **hist_params
             )
    
    E_ax.text(0.75, 0.55, lbl1, transform=E_ax.transAxes)
    pT_ax.text(0.75, 0.55, lbl1, transform=pT_ax.transAxes)
    eta_ax.text(0.75, 0.55, lbl1, transform=eta_ax.transAxes)   
    
    pT_ax.hist(part_pT,
               range=[10,1000],
               **hist_params
              )
        
    eta_ax.hist(part_eta,
                range=[-4,4],
                **hist_params
                )

    tick_param={'direction':'in',
                'length':6,
                'width':1.25
               }
    
    E_ax.tick_params(axis='both',
                     **tick_param)
    pT_ax.tick_params(axis='both',
                      **tick_param)
    eta_ax.tick_params(axis='both',
                       **tick_param)

    E_ax.legend()
    pT_ax.legend()
    eta_ax.legend()
    
    plt.tight_layout()
    
    return 

###########################################
def Dist_plot(p1, p2, *args):
    
    figsize=args[0]
    lbl1=args[1]
    range_E=args[2]
    range_pT=args[3]
	
    fig = plt.figure(figsize=figsize)
    mpl.rcParams['font.family'] = 'Ubuntu Mono'
    layout = (1, 3)
    
    E_ax   = plt.subplot2grid(layout, (0, 0))
    E_ax.set_xlabel('Energy (GeV)')
    E_ax.set_ylabel('Normalized Events')
    
    pT_ax  = plt.subplot2grid(layout, (0, 1))
    pT_ax.set_xlabel('Tranverse Momentum (GeV)')
    
    eta_ax = plt.subplot2grid(layout, (0, 2))
    eta_ax.set_xlabel('Rapidity')

    hist_params_sig={'histtype':'step',
                     'bins':50,
                     'normed':True,
                     'label':'Signal'}

    hist_params_tot={'histtype':'step',
                     'bins':50,
                     'normed':True,
                     'label':'Total'}
    
    E_ax.hist(p1['E'], 
              range=range_E,
              **hist_params_sig
             )
    
    E_ax.hist(p2['E'], 
              range=range_E,
              **hist_params_tot
             )
    
    E_ax.text(0.75, 0.55, lbl1, transform=E_ax.transAxes)
    pT_ax.text(0.75, 0.55, lbl1, transform=pT_ax.transAxes)
    eta_ax.text(0.75, 0.55, lbl1, transform=eta_ax.transAxes)   
    
    pT_ax.hist(p1['pT'],
               range=range_pT,
               **hist_params_sig
              )
    
    pT_ax.hist(p2['pT'],
               range=range_pT,
               **hist_params_tot
              )
        
    eta_ax.hist(p1['Eta'],
                range=[-4,4],
                **hist_params_sig
                )

    eta_ax.hist(p2['Eta'],
                range=[-4,4],
                **hist_params_tot
                )

    tick_param={'direction':'in',
                'length':6,
                'width':1.25
               }
    
    E_ax.tick_params(axis='both',
                     **tick_param)
    pT_ax.tick_params(axis='both',
                      **tick_param)
    eta_ax.tick_params(axis='both',
                       **tick_param)

    E_ax.legend()
    pT_ax.legend()
    eta_ax.legend()
    
    plt.tight_layout()
    
    return 
