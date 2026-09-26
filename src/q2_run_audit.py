"""Read-only full-run integrity, independent formula and integral audit for Q2.

No solver, clipping, rounding, or alteration of a computed trajectory.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits, threadpool_info
from q2_full_analysis import BASE, ROOT, load_main, maximum, serial
from q2_model import sha256


def audit(label):
    d=load_main(label); meta=d['_meta']; n=meta['signature']['settings']['cells']
    radius=.02; length=.25; dr=radius/n
    # Independent analytic volumes, not the production difference of squared faces.
    v=np.pi*length*(2*np.arange(n)+1)*dr**2
    volume=np.pi*length*radius**2
    assert abs(v.sum()-volume)<1e-18
    ts=d['time_s']; rr=d['radius_m']
    integer=(ts==np.floor(ts))
    assert np.array_equal(ts[integer],np.arange(10801))
    assert np.array_equal(rr,np.arange(21)*.001)
    assert np.all(d['T_C'][0]==28) and np.all(d['C_kg_kg'][0]==2.55)
    assert np.all(np.isfinite(d['T_C'])) and np.all(np.isfinite(d['C_kg_kg']))
    assert np.all(d['C_kg_kg']>0) and np.all(d['T_C']+273.15>0)
    assert len(meta['segments'])==180
    from openpyxl import load_workbook
    wb=load_workbook(ROOT/'A题/附件/附件1.xlsx',read_only=True,data_only=True)
    rows=list(wb[wb.sheetnames[0]].iter_rows(min_row=2,values_only=True));wb.close()
    env=np.array([[r[0],r[1],r[2]] for r in rows],dtype=float)
    assert np.array_equal(env[:,0],np.arange(0,14401,60))
    ta=np.interp(ts,env[:,0],env[:,1]);ce=np.interp(ts,env[:,0],env[:,2])
    # Evaluated interpolation cannot be outside the measured coverage.
    assert ts[0]==0 and ts[-1]==10800 and ts[-1]<=env[-1,0]
    gx,gw=leggauss(32); maxes=dict(C_integral_m3_C=0.,heat_chain_J=0.,
                                  halfcell_heat_W_m2=0.,halfcell_moisture_m_s=0.,
                                  surface_root_equivalent_C=0.)
    w0=(650+128*2.55)*(1450+2736*2.55/(1+2.55))
    previous=0.
    for seg in meta['segments']:
        assert seg['start_s']==previous and seg['end_s']==previous+60
        p=BASE/'runs'/label/seg['file']; assert sha256(p)==seg['sha256']
        with np.load(p) as q:
            y=q['state']; assert y.shape==(2*n+3,) and np.isfinite(y).all()
            assert q['end_s']==seg['end_s']
            T=y[:2*n:2];C=y[1:2*n:2]
            W=(650+128*C)*(1450+2736*C/(1+C))
            cm=abs(np.dot(v,C-2.55)+volume*y[-3])
            hm=abs(np.dot(v,W*(T-28))-volume*w0*(y[-2]+y[-1]))
            maxes['C_integral_m3_C']=max(maxes['C_integral_m3_C'],float(cm))
            maxes['heat_chain_J']=max(maxes['heat_chain_J'],float(hm))
            it=int(np.flatnonzero(ts==seg['end_s'])[0])
            Ts,Cs=float(d['T_C'][it,-1]),float(d['C_kg_kg'][it,-1])
            k0=.21+.38*C[-1]/(1+C[-1]);ks=.21+.38*Cs/(1+Cs)
            kb=2*k0*ks/(k0+ks)
            a0=np.exp(-3850/(T[-1]+273.15));a1=np.exp(-3850/(Ts+273.15))
            ab=2*a0*a1/(a0+a1)
            # Independent 32-point quadrature of b, on the short surface chord.
            dc=C[-1]-Cs;nodes=(C[-1]+Cs)/2+dc*gx/2
            psi_difference=dc/2*np.dot(gw,np.exp(-.45/nodes))
            ht=kb*(T[-1]-Ts)/(dr/2)-25*(Ts-ta[it])
            hc=.0024*ab*psi_difference/(dr/2)-8e-7*(Cs-ce[it])
            scale=.0024*ab*np.exp(-.45/Cs)/(dr/2)+8e-7
            maxes['halfcell_heat_W_m2']=max(maxes['halfcell_heat_W_m2'],float(abs(ht)))
            maxes['halfcell_moisture_m_s']=max(maxes['halfcell_moisture_m_s'],float(abs(hc)))
            maxes['surface_root_equivalent_C']=max(maxes['surface_root_equivalent_C'],float(abs(hc/scale)))
            if seg['end_s']==10800:assert np.array_equal(d['final_state'],y)
        previous=seg['end_s']
    maxes['C_integral_normalized_to_initial']=maxes['C_integral_m3_C']/(volume*2.55)
    maxes['heat_chain_normalized_by_initial_J_per_K']=maxes['heat_chain_J']/(volume*w0)
    assert maxes['C_integral_normalized_to_initial']<1e-8
    assert maxes['heat_chain_normalized_by_initial_J_per_K']<1e-7
    lower_T=np.minimum.accumulate(np.minimum(ta,28))
    upper_T=np.maximum.accumulate(np.maximum(ta,28))
    lower_C=np.minimum.accumulate(np.minimum(ce,2.55))
    upper_C=np.maximum.accumulate(np.maximum(ce,2.55))
    ranges={}
    for f,lo,hi in [('T_C',lower_T,upper_T),('C_kg_kg',lower_C,upper_C)]:
        a=d[f]; excess=np.maximum(lo[:,None]-a,a-hi[:,None])
        ranges[f]=dict(min=float(a.min()),max=float(a.max()),
            largest_history_range_excess=float(max(0,excess.max())))
        assert excess.max()<1e-9
    return dict(label=label,question='Q2',status='RUN_INTEGRITY_AND_EQUATION_LEDGER_PASS',
        source=d['_source'],source_sha256=meta['solution_sha256'],
        audit_threadpools=threadpool_info(),auditor_sha256=sha256(__file__),
        checked_complete_checkpoints=180,integer_values=453600,
        original_initial_exact=True,final_state_matches_last_checkpoint=True,
        no_output_extrapolation=True,no_clipping=True,independent_checkpoint_audit=maxes,
        ranges=ranges,surface_minus_environment=dict(
            T_min=float(np.min(d['T_C'][:,-1]-ta)),T_max=float(np.max(d['T_C'][:,-1]-ta)),
            C_min=float(np.min(d['C_kg_kg'][:,-1]-ce)),C_max=float(np.max(d['C_kg_kg'][:,-1]-ce))),
        actual_run_stats=meta['stats'],wall_s=meta['active_wall_s'],
        numerical_accuracy_pass_implied=False,
        limitations=['Checkpoint recomputation independently checks formulas/geometry and auxiliary integral bookkeeping, not the continuum solution error.',
            'Volume times dry-basis C is not kg of water. The heat ledger includes the W-prime chain term and does not add latent heat.',
            'Root-equivalent C here uses the positive local concentration scale; this does not estimate halfcell spatial error.',
            'History-range check covers actual stored times; no claim about all continuous times.',
            'Tiny floating-point range excesses are reported, never clipped.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args()
    with threadpool_limits(limits=1):
        result=audit(args.label)
    folder=BASE/'run-audits';folder.mkdir(exist_ok=True)
    path=folder/(args.label+'.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        old_id=old.get('auditor_sha256','before-explicit-audit-thread-pin')
        archive=folder/(args.label+'-'+old_id[:20]+'.json')
        if not archive.exists():archive.write_bytes(path.read_bytes())
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=serial),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False,default=serial))


if __name__=='__main__':main()
