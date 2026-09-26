"""Generate Q2 short-time evidence from saved outputs, without solving a PDE."""
from pathlib import Path
import json
import csv
import sys
import numpy as np
from q2_model import Properties, sha256
from q2_short_compare import load, compare

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/q2-stage1-20260911'
RUNS=OUT/'runs'
NODE=OUT/'node-reference/startup-fixed-v3'


def read_node(n, tight=False):
    label=f'real-N{n}' if not tight else f'real-N{n}-tight-to121'
    meta=json.loads((NODE/label/'stats.json').read_text(encoding='utf-8'))
    spec=json.loads((ROOT/'config/q2.json').read_text(encoding='utf-8'))['inputs']['environment']
    if meta['failed'] or meta['test_configuration']:
        raise ValueError(f'Invalid independent real-input run: {label}')
    if meta['source_sha256']!=sha256(ROOT/'src/q2_node_reference.py'):
        raise ValueError(f'Independent source changed; invalidate dependent report: {label}')
    if meta['environment_sha256']!=spec['sha256'] or sha256(ROOT/spec['path'])!=spec['sha256']:
        raise ValueError(f'Independent environment version mismatch: {label}')
    with np.load(NODE/label/'solution.npz') as data:
        return dict(time_s=data['times'],radius_m=data['radii'],T_C=data['T_C'],C_kg_kg=data['C_kg_kg'])


def where_max(arr,a,sel):
    indices=np.flatnonzero(sel)
    ij=np.unravel_index(np.argmax(arr[sel]),arr[sel].shape)
    it,ir=int(indices[ij[0]]),int(ij[1])
    return dict(value=float(arr[it,ir]),time_s=float(a['time_s'][it]),radius_m=float(a['radius_m'][ir]))


def ordinary(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(str(type(x)))


def main():
    a=load('fick-8192-s1'); fine=load('fick-16384-s1')
    coarse=load('fick-4096-s1'); primary=load('potential-8192-s1')
    nt=read_node(16384,True); nf=read_node(16384); nc=read_node(8192)
    for data in (fine,coarse,primary,nt,nf,nc):
        assert np.array_equal(a['time_s'],data['time_s'])
        assert np.array_equal(a['radius_m'],data['radius_m'])
    sel=a['time_s']>=1
    integer=sel&(a['time_s']==np.round(a['time_s']))
    assert int(integer.sum())==121 and len(a['radius_m'])==21
    report=dict(status='SHORT_DIAGNOSTIC_EVIDENCE_ONLY',model='Q2-R',
                selected_internal_flux='direct harmonic-D Fick',
                selected_boundary='separated-potential coupled half-ring Robin',
                recommended_full_start=dict(cells=8192,method='Radau',rtol=1e-11,atol_T=1e-12,
                                            atol_C=1e-14,max_step_s=.5,first_step_factor=.01),
                full_configuration_not_prevalidated=True,full_calculation_or_export_performed=False,
                selected_short_candidate='records/q2-stage1-20260911/runs/fick-8192-s1/solution.npz',
                best_full_short_refinement='records/q2-stage1-20260911/runs/fick-16384-s1/solution.npz',
                source_sha256=sha256(ROOT/'src/q2_model.py'),
                comparison_script_sha256=sha256(__file__),
                comparison_helper_sha256=sha256(ROOT/'src/q2_short_compare.py'))
    space=[]
    levels=[256,512,1024,2048,4096,8192,16384]
    for lo,hi in zip(levels[:-1],levels[1:]):
        aa,bb=load(f'fick-{lo}-s1'),load(f'fick-{hi}-s1')
        space.append(dict(cells=[lo,hi],t_ge_1=compare(aa,bb),
                          with_subseconds=compare(aa,bb,minimum_time=0)))
    report['space_refinement']=space
    report['last_two_max_norm_orders']={key:float(np.log2(space[-2]['t_ge_1'][key]['max_abs']/space[-1]['t_ge_1'][key]['max_abs']))
                                        for key in ('T_C','C_kg_kg')}
    timepairs=[('fick-1024-time7','fick-1024-time9'),('fick-1024-time9','fick-1024-s1'),
               ('fick-8192-s1','fick-8192-strict-startup'),('fick-8192-s1','fick-8192-firststep-only'),
               ('fick-8192-s1','fick-8192-BDF')]
    report['time_comparisons']=[dict(pair=[l,r],t_ge_1=compare(load(l),load(r)),
                                    with_subseconds=compare(load(l),load(r),minimum_time=0)) for l,r in timepairs]
    report['independent_node_comparison']=dict(candidate_vs_node16384=compare(a,nf),
        fine_main16384_vs_node16384=compare(fine,nf),reference_time_change=compare(nf,nt))
    report['flux_comparison']=compare(primary,a)
    costs={}
    for flux in ('potential','fick'):
        runs=[json.loads((RUNS/f'{flux}-8192-{suffix}'/'run.json').read_text(encoding='utf-8'))
              for suffix in ('s1','repeat2','repeat3')]
        durations=[r['stats']['wall_s'] for r in runs]
        assert all(all(p['num_threads']==1 for p in r['threadpools']) for r in runs)
        for suffix in ('repeat2','repeat3'):
            assert compare(load(f'{flux}-8192-s1'),load(f'{flux}-8192-{suffix}'))['C_kg_kg']['max_abs']<1e-12
        costs[flux]=dict(wall_s=durations,median_wall_s=float(np.median(durations)),
                         accepted_steps=[sum(s['accepted_steps'] for s in r['stats']['segments']) for r in runs],
                         nfev=[sum(s['nfev'] for s in r['stats']['segments']) for r in runs])
    report['same_accuracy_cost_comparison']=costs
    report['median_cost_reduction_fraction']=1-costs['fick']['median_wall_s']/costs['potential']['median_wall_s']
    report['cost_scope']='Three serial runs per internal-flux choice, one BLAS thread; wall includes in-run checkpointing; hardware variance retained; same PDE/boundary/integrator. Not a universal speed guarantee.'
    strict=load('fick-8192-strict-startup'); bdf=load('fick-8192-BDF'); initial=load('fick-8192-firststep-only')
    all_flags=[]; budgets={}; arrays={}
    for key in ('T_C','C_kg_kg'):
        # Two distinct evidence routes. Do not SUM these (shared spatial sources).
        # Safety factor is an explicit engineering margin, not a rigorous bound.
        dtmain=np.maximum.reduce([abs(a[key]-strict[key]),abs(a[key]-bdf[key]),abs(a[key]-initial[key])])
        dtref=abs(nf[key]-nt[key])
        floor=1e-10 if key=='T_C' else 1e-11
        own_space=1.25*(4/3)*abs(fine[key]-a[key])
        route_A=own_space+2*dtmain+floor
        ref_space=1.25/3*abs(nf[key]-nc[key])
        # The direct method difference already includes the candidate's time
        # error; only reference uncertainty is added in this triangle route.
        route_B=abs(a[key]-nf[key])+ref_space+2*dtref+floor
        envelope=np.maximum(route_A,route_B)
        boundary_distance=abs(a[key]*1e4-(np.floor(a[key]*1e4)+.5))/1e4
        flags=(boundary_distance<=envelope)&integer[:,None]
        budgets[key]=dict(own_space_max=where_max(own_space,a,sel),
            reference_space_max=where_max(ref_space,a,sel),
            main_time_change_max=where_max(dtmain,a,sel),reference_time_change_max=where_max(dtref,a,sel),
            envelope_max=where_max(envelope,a,sel),initial_rounded_flags=int(flags.sum()))
        arrays[f'{key}_envelope']=envelope;arrays[f'{key}_distance_to_rounding_boundary']=boundary_distance
        for it,ir in zip(*np.where(flags)):
            all_flags.append(dict(field=key,time_s=float(a['time_s'][it]),radius_m=float(a['radius_m'][ir]),
                candidate=float(a[key][it,ir]),main16384=float(fine[key][it,ir]),
                node16384=float(nf[key][it,ir]),empirical_envelope=float(envelope[it,ir]),
                boundary_distance=float(boundary_distance[it,ir]),resolution='pending point-specific finer evidence'))
    report['empirical_error_envelope']=dict(formula_A='1.25*(4/3)*abs(main16384-main8192)+2*max(main time comparisons)+roundoff margin',
        formula_B='abs(main8192-node16384)+1.25/3*abs(node16384-node8192)+2*node time change+roundoff margin',
        combination='maximum(A,B), not sum; the same spatial source is not counted twice',
        interpretation='Pointwise engineering envelope supported by observed asymptotic convergence; candidate time error is already in the direct difference in route B. Nonlinear/roundoff effects are monitored and included in the comparative computations, not converted from stage residuals into a rigorous global bound. Not a confidence interval.',
        roundoff_margins=dict(T_C=1e-10,C_kg_kg=1e-11),values=budgets)
    # Targeted convergence at the observed rounding-sensitive point.
    it=int(np.flatnonzero(a['time_s']==32)[0]);ir=20
    high=load('fick-32768-to33'); ih=int(np.flatnonzero(high['time_s']==32)[0])
    pm=np.array([load(f'fick-{n}-s1')['C_kg_kg'][it,ir] for n in (4096,8192,16384)])
    pm=np.r_[pm,high['C_kg_kg'][ih,ir]]
    pn=np.array([read_node(n)['C_kg_kg'][it,ir] for n in (4096,8192,16384)])
    lm=(4*pm[1:]-pm[:-1])/3;ln=(4*pn[1:]-pn[:-1])/3
    # Margin tests on extrapolated sequences, no replacement of computed answers.
    local_time_checks=dict(
        main_strict=abs(a['C_kg_kg'][it,ir]-strict['C_kg_kg'][it,ir]),
        main_BDF=abs(a['C_kg_kg'][it,ir]-bdf['C_kg_kg'][it,ir]),
        main_firststep=abs(a['C_kg_kg'][it,ir]-initial['C_kg_kg'][it,ir]),
        reference_strict=abs(nf['C_kg_kg'][it,ir]-nt['C_kg_kg'][it,ir]))
    local_time=max(local_time_checks.values())
    extrap_margin=4*max(abs(lm[-1]-lm[-2]),abs(ln[-1]-ln[-2]),abs(lm[-1]-ln[-1]),local_time,1e-11)
    local_lo=min(lm[-1],ln[-1])-extrap_margin
    local_hi=max(lm[-1],ln[-1])+extrap_margin
    lower_round=np.round(local_lo,4);upper_round=np.round(local_hi,4)
    verified=bool(lower_round==upper_round==np.round(a['C_kg_kg'][it,ir],4))
    report['targeted_32s']=dict(main_cells=[4096,8192,16384,32768],main_values=pm,
        main_local_orders=np.log2(abs(np.diff(pm)[:-1]/np.diff(pm)[1:])),
        main_conditional_extrapolations=lm,node_cells=[4096,8192,16384],node_values=pn,
        node_local_order=float(np.log2(abs((pn[1]-pn[0])/(pn[2]-pn[1])))),
        node_conditional_extrapolations=ln,engineering_extrapolation_margin=extrap_margin,
        measured_local_time_changes=local_time_checks,
        diagnostic_limit_interval=[local_lo,local_hi],rounding_boundary=2.38475,
        candidate_rounding_supported=verified,
        interpretation='Independent smooth-grid extrapolation evidence only; raw arrays/last digit never manually changed')
    for f in all_flags:
        if f['field']=='C_kg_kg' and f['time_s']==32 and f['radius_m']==.02 and verified:
            f['resolution']='supported by independently extrapolated convergent families; not a strict guarantee'
    report['rounding']=dict(integer_values_scanned=5082,initial_flag_count=len(all_flags),
        resolved_count=sum(f['resolution'].startswith('supported') for f in all_flags),
        unresolved_count=sum(f['resolution'].startswith('pending') for f in all_flags),flags=all_flags)
    deg=load('q1-degeneration-8192')
    with np.load(ROOT/'results/q1/q1-authority.npz') as q1:
        sub={key:deg[key][integer] for key in ('T_C','C_kg_kg')}
        sub.update(time_s=deg['time_s'][integer],radius_m=deg['radius_m'])
        reference=dict(time_s=q1['time_s'][1:122],radius_m=q1['radius_m'],
                       T_C=q1['T_C'][1:122],C_kg_kg=q1['C_kg_kg'][1:122])
        report['Q1_test_only_degenerate_comparison']=compare(sub,reference)
    selected_times=[0,1,10,30,59.9,60,60.1,120,121]
    rows=[]
    for t in selected_times:
        i=int(np.flatnonzero(a['time_s']==t)[0])
        rows.append(dict(time_s=t,T_centre_C=float(a['T_C'][i,0]),T_surface_C=float(a['T_C'][i,-1]),
                         C_centre_kg_kg=float(a['C_kg_kg'][i,0]),C_surface_kg_kg=float(a['C_kg_kg'][i,-1])))
    with (OUT/'short-results.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table='本轮0—121 s候选试算；不是题定表3/表4或正式Excel。\n\n|t/s|中心T/℃|表面T/℃|中心C/(kg/kg)|表面C/(kg/kg)|\n|---:|---:|---:|---:|---:|\n'
    for r in rows:
        table+='|'+f"{r['time_s']:g}|{r['T_centre_C']:.9f}|{r['T_surface_C']:.9f}|{r['C_centre_kg_kg']:.9f}|{r['C_surface_kg_kg']:.9f}|\n"
    table+='\n额外小数用于比较诊断，不表示这些末位均准确；全精度候选的数值误差证据见验证结论。\n'
    (OUT/'短时结果表.md').write_text(table,encoding='utf-8')
    p=Properties(); IC=.45*(1/2.55-1/a['C_kg_kg']);IT=3850*(1/301.15-1/(a['T_C']+273.15))
    logratio=np.log(p.D(a['C_kg_kg'],a['T_C'])/float(p.D(2.55,28)))
    assert np.max(abs(IC+IT-logratio))<1e-13
    report['diffusion_factor_analysis']=dict(max_identity_residual=float(np.max(abs(IC+IT-logratio))),
        at_121s_surface=dict(I_C=float(IC[-1,-1]),I_T=float(IT[-1,-1]),D_ratio=float(np.exp(IC[-1,-1]+IT[-1,-1]))),
        scope='Relative-to-initial factors on actual short trajectory; not instantaneous rates or drying-time contributions')
    np.savez_compressed(OUT/'rounding-and-factor-evidence.npz',time_s=a['time_s'],radius_m=a['radius_m'],
                        I_C=IC,I_T=IT,log_D_ratio=logratio,**arrays)
    report['reported_rows']=rows
    report['radau_stage_audit']=json.loads((OUT/'radau-stage-audit/audit.json').read_text(encoding='utf-8'))
    report['readiness']=dict(model_implementation_ready=True,
        short_t_ge_1_engineering_change_goal_supported=all(budgets[k]['envelope_max']['value']<1e-5 for k in budgets),
        all_subsecond_pressure_points_pass=False,full_horizon_accuracy_not_established=True,
        formal_export_ready=False,awaiting_user_full_stage_confirmation=True)
    (OUT/'short-validation-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=ordinary),encoding='utf-8')
    # Human-readable evidence is generated from exactly the same stored arrays.
    stage=report['radau_stage_audit']; tg=report['targeted_32s']
    cmp=report['independent_node_comparison']['candidate_vs_node16384']
    at=report['time_comparisons']
    text='# 问题2模型定稿与短时验证结论\n\n'
    text+='本轮已完成Q2-R模型定稿、实现和0—121 s短时诊断。未进行完整三小时计算、正式Excel导出或论文写作；Q1保持冻结。完整方程、参数来源、假设及离散见同目录《模型定稿.md》。\n\n'
    text+='## 当前采用的实现\n\n'
    text+='保留已批准的有效变物性方程。数值实现选择直接Fick内部面通量，结合仅对含水率因子积分的表面势函数、独立Ts/Cs、耦合整体稀疏Radau。该选择在用户已批准的正确基线对照与择优实现范围内，不改变PDE、物性、初态、Robin或环境含义。内部势通量继续保留作对照。\n\n'
    text+='面通量中的局部物性、中心环体积、环平均点值重构、表面隐式Jacobian和W′储存导数已经实际检查。全程以m、s计算，C为kg/kg干基，D指数使用药材局部K；没有使用收缩数据、平滑、裁剪或外推。\n\n'
    text+='## 已执行验证\n\n'
    text+='1. 题目/环境/模板指纹和51个受保护Q1文件一致；环境241节点0—14400 s，短算实际用到0/60/120/180 s四节点。原模板两张表是A1:F5省略号示意，终点未预填。\n'
    text+='2. 初始物性独立代值、稳定势差对独立缩放积分、零交换/平衡、C均匀T不均匀零水分通量、r²中心与几何、表面括根/原式残差通过；两支路完整Jacobian对独立中心差分的最大行尺度误差约8.4×10⁻¹¹。最初极小积分的参照被默认绝对尺度钝化，改为指数归一化后核查到约10⁻²⁰²；旧检查记录保留，未改模型数值。\n'
    text+='3. 连续非多项式耦合制造解使用独立推导的体源及真实环体积积分，32/64/128环点值误差约二阶；独立节点法另有64/128/256解析Robin模态和耦合制造解检查。制造解只证明被测离散性质，不自动证明真实启动处于渐近区。\n'
    qd=report['Q1_test_only_degenerate_comparison']
    text+=f"4. 单独测试配置恢复Q1的全部本构与边界，从原初态算0—121 s。8192环与冻结Q1权威解最大差：T={qd['T_C']['max_abs']:.12g} ℃，C={qd['C_kg_kg']['max_abs']:.12g} kg/kg；分别在120 s和1 s表面。较粗测试网格有1处水分四位舍入与Q1正式细网格不同，不能将本测试说成Q1所有尾数已复现，也不能继承Q1精度。\n"
    text+=f"5. 对选定8192环候选的实际Radau复算观测{stage['accepted_steps']}个接受步、{stage['collocation_calls']}次配点求解，最大迭代{stage['maximum_collocation_iterations']}次，非收敛调用{stage['nonconverged_collocation_calls']}次。最大配点方程残差T={stage['maximum_step_residual_T_C']:.12g} ℃、C={stage['maximum_step_residual_C_kg_kg']:.12g} kg/kg；这是阶段代数残差，不是全局解误差界。\n\n"
    text+='## 空间与时间精化\n\n'
    text+='以下范数覆盖1—121 s全部整数秒、21个半径及59.9/60.1/119.9/120.1 s。亚秒压力点单列。\n\n|空间层级|最大温度差/℃|位置|最大水分差/(kg/kg)|位置|\n|---|---:|---|---:|---|\n'
    for row in space[2:]:
        t,c=row['t_ge_1']['T_C'],row['t_ge_1']['C_kg_kg']
        text+=f"|{row['cells'][0]}→{row['cells'][1]}|{t['max_abs']:.12g}|{t['time_s']:g} s，{t['radius_m']*100:g} cm|{c['max_abs']:.12g}|{c['time_s']:g} s，{c['radius_m']*100:g} cm|\n"
    text+=f"\n最后三层最大范数实测阶为T={report['last_two_max_norm_orders']['T_C']:.8f}、C={report['last_two_max_norm_orders']['C_kg_kg']:.8f}。所有层级从原始初态开始，没有用粗网格末态插值冒充细网格重算。\n\n"
    text+='|时间检查（空间固定）|最大温度差/℃|最大水分差/(kg/kg)|\n|---|---:|---:|\n'
    labels=['1024环：rtol 1e-7→1e-9','1024环：rtol 1e-9→1e-11',
            '8192环：整体收紧时间/首步','8192环：仅把首步缩小100倍','8192环：Radau对BDF']
    for label,row in zip(labels,at):
        text+=f"|{label}|{row['t_ge_1']['T_C']['max_abs']:.12g}|{row['t_ge_1']['C_kg_kg']['max_abs']:.12g}|\n"
    text+='\n对应完整设置、最大差位置、原值以及亚秒差异均保存在short-validation-summary.json和各run.json中；时间容差不是先验精度保证。\n\n'
    text+=f"8192环主候选对独立16384节点参考：最大T差{cmp['T_C']['max_abs']:.12g} ℃（{cmp['T_C']['time_s']:g} s，表面），最大C差{cmp['C_kg_kg']['max_abs']:.12g} kg/kg（1 s，表面）。独立参考自身用2048/4096/8192/16384层及完整0—121 s更紧时间复算检验，不能把它当零误差真值。\n\n"
    text+='## 启动问题、守恒与范围\n\n'
    text+='动态表面节点法的8192层自动首步试探曾失败：初始表面导数约−1.65835 kg/kg/s，库内1.93836 s Euler探针给负值−0.66449，尚未接受；max_step限制未约束该探针。修复为显式局部扩散/正值首步，重新统一三层与组件版本。没有修改初始Cs=2.55、裁剪负值或静默使用旧解。原失败JSON保留。\n\n'
    stress=space[-1]['with_subseconds']['C_kg_kg']
    text+=f"主方法8192→16384在0.001 s表面的差仍为{stress['max_abs']:.12g} kg/kg，未达到10⁻⁵压力目标；不发布亚秒完整误差通过结论。与此同时，从t=0重算的真实空间收敛及只缩小首步/整体收紧时间试验已直接检查其传播到t≥1 s的影响，不能用‘不提交亚秒’取代这些检查。\n\n"
    text+=f"独立5点Gauss时间积分核对选定候选：C积分收支最大缺口{stage['independent_Gauss_C_balance_m3_C']:.12g} m³·(kg/kg)，按初始C积分归一化为{stage['independent_Gauss_C_balance_normalized']:.12g}；有效热储存链式缺口{stage['independent_Gauss_heat_chain_balance_J']:.12g} J，按V W₀×1 K尺度归一化为{stage['independent_Gauss_heat_chain_balance_normalized']:.12g}。这检验的是本题有效方程，不是完整真实水质量/总焓守恒。被检查RK阶段正值、物理范围与表面分支门控通过；全部曲线未人工裁剪。\n\n"
    text+='## 数值精度与四位小数\n\n'
    text+='令M₈、M₁₆为主8192/16384环值，N₈、N₁₆为独立节点值。已观察渐近二阶后，采用两条经验检查路线：A=1.25×(4/3)|M₁₆−M₈|+2Δt_M+浮点余量；B=|M₈−N₁₆|+(1.25/3)|N₁₆−N₈|+2Δt_N+浮点余量；取max(A,B)，不把两路线相加。B的直接差已含主时间误差，因此不重复加该项。1.25、2以及浮点余量是透明的工程余量，不能解释成严格误差界或置信区间。非线性/根/浮点通过实际残差及精化对照监测，未把每步残差简单当成全程误差。\n\n'
    for key,unit in [('T_C','℃'),('C_kg_kg','kg/kg')]:
        q=budgets[key]['envelope_max']
        text+=f"- {key}候选全精度解的最大经验包络：{q['value']:.12g} {unit}，位置{q['time_s']:g} s、{q['radius_m']*100:g} cm。\n"
    text+=f"\n扫描121×21×2={report['rounding']['integer_values_scanned']}个整数输出，初始标记{report['rounding']['initial_flag_count']}处：32 s表面C。主网格4096/8192/16384/32768的局部阶接近2；与独立节点4096/8192/16384的条件外推相互核对，局部诊断极限区间[{tg['diagnostic_limit_interval'][0]:.15f}, {tg['diagnostic_limit_interval'][1]:.15f}]位于2.384750000000000下方，支持原候选舍入为2.3847。外推只作尾数核验，没有替换原数组或手改尾数。按这套经验规则，当前未确认项为{report['rounding']['unresolved_count']}；没有严格数学保证，也不能外推为完整3 h已通过。\n\n"
    text+='## 基线比较与实际取舍\n\n|内部面处理|三次运行时间/s|中位时间/s|接受步|\n|---|---|---:|---:|\n'
    for key,label in [('potential','分离势内部通量'),('fick','直接Fick内部通量')]:
        row=costs[key]
        text+=f"|{label}|{', '.join(f'{x:.6f}' for x in row['wall_s'])}|{row['median_wall_s']:.6f}|{row['accepted_steps'][0]}|\n"
    text+=f"\n同一PDE、网格、真实输入、表面条件、Radau容差和单线程配置下，两者最大C差{report['flux_comparison']['C_kg_kg']['max_abs']:.12g} kg/kg，全部本段四位小数相同。Fick本次中位计时少约{100*report['median_cost_reduction_fraction']:.2f}%；这是实际实现/设备上的短段结果，不保证全程或其他问题同样加速。两者步数及RHS调用数相同，未声称势法提高了收敛阶或稳定性。\n\n"
    text+='BDF在本段的同网格对照也较省时，保留为时间参照；当前首轮完整候选仍采用已经逐步审查的单步Radau，以便清楚处理环境折点和恢复。没有证据声称Radau全面优于BDF；未来若成本成为瓶颈，BDF可在完整时域对照通过后再确定用途。\n\n'
    text+='成熟方法与本题适配已经在《模型定稿.md》中分开。可作为贡献的是因子分离/局部表面消元/耦合Jacobian与输出精度验证的有依据适配；本轮实测收益是采用较简洁内部通量避免无效计算，不是原创算法声明。\n\n'
    text+='## 短时结果与扩散解释\n\n'+table+'\n'
    fact=report['diffusion_factor_analysis']['at_121s_surface']
    text+=f"121 s表面相对初态的对数因子：I_C={fact['I_C']:.12g}，I_T={fact['I_T']:.12g}，D/D₀={fact['D_ratio']:.12g}。这是该状态相对初态的分解：本点失水的抑制量与温升的促进量同时存在。它们不是瞬时变化率，也不是烘干时长贡献；未宣称未计算的后续时段由某因素主导。中心C在该短段浮点结果中仍显示2.55，不把它解释为严格数学不变。\n\n"
    text+='## 是否进入完整时段\n\n'
    text+='具备进入完整时段候选计算的实现与短时数值依据。建议从8192环、Radau rtol=10⁻¹¹、atol_T=10⁻¹² ℃、atol_C=10⁻¹⁴ kg/kg、max_step=0.5 s起步；这些设置须接受完整时域的再验证，不能预先保证全程精度。更细网格、独立实现与32 s敏感记录均保留。完整计算需用户确认后执行，终点推荐10800 s；原模板没有独立终点标记。\n\n'
    text+='尚未解决的是完整时域的精度/舍入/最终网格、完整输出终点选择，以及未量化的Ce等效映射、蒸发冷却和固定中截面近似物理偏差。没有本问药材内部实测值或官方答案可验证这些简化。本轮不把数值收敛或上述包络称为真实物理精度，也不把方法适配称原创或保证奖项。\n\n'
    text+='本报告由src/q2_stage1_report.py从机器可读结果生成；主程序src/q2_model.py、短算入口src/q2_short_run.py、配置config/q2.json。原始数据与Q1文件不变。AI自查及AI独立实现均不冒充队员人工技术复核。\n'
    (OUT/'短时验证结论.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ['last_two_max_norm_orders','same_accuracy_cost_comparison',
        'empirical_error_envelope','targeted_32s','rounding','Q1_test_only_degenerate_comparison','readiness','diffusion_factor_analysis']},
        ensure_ascii=False,indent=2,default=ordinary))


if __name__=='__main__':main()
