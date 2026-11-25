import csv, sys, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
"""Sensitivity analyses for the revision (reviewer requests): first vs. minimum candidate,
deduplication, rebuilt Human->Human reference, cluster bootstrap, smoothing, JSD/TV, overlap.

Usage: python sensitivity_analyses.py [release_root] > ../results/sensitivity_results.json
Requires the FULL source texts. The public release redacts 75 passages (news, chlaep.org.uy);
the reported numbers were computed from the full texts (private archive). With the redacted
release the script stops with an error.
"""
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kldi import _tokens
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
rd=lambda p:list(csv.DictReader(open(R/p,encoding="utf-8")))
src={r["document_id"]:r for r in rd("data/source_passages.csv")}
par={r["document_id"]:r["paraphrased_text"] for r in rd("data/selected_paraphrases.csv")}
ha=rd("results/human_ai_scores.csv"); hh=rd("data/human_human_pairs.csv")
def auc_inv(hh_s,ha_s):  # P(HA < HH) i.e. inverted orientation AUC (HA positive, smaller=positive)
    y=[0]*len(hh_s)+[1]*len(ha_s); return 1-roc_auc_score(y,list(hh_s)+list(ha_s))
def tuk(x): q1,q3=np.percentile(x,[25,75]); return q1,q3,q3-q1,q3+1.5*(q3-q1)
def summ(x): x=np.array(x); q1,q3,i,t=tuk(x); return dict(n=len(x),min=x.min(),q1=q1,med=np.median(x),q3=q3,max=x.max(),iqr=i,T=t)
fmt=lambda d:{k:(round(float(v),3) if k!="n" else int(v)) for k,v in d.items()}
out={}
hk=np.array([float(r["kldi"]) for r in ha]); hhs=np.array([float(r["kldi"]) for r in hh])
out["HH"]=fmt(summ(hhs)); out["HA"]=fmt(summ(hk))
out["AUC_inv_main"]=round(auc_inv(hhs,hk),3)
# (b) first vs best
ini=np.array([float(r["kldi_initial"]) for r in ha]); best=np.array([float(r["kldi_best"]) for r in ha])
out["first_vs_min"]=dict(auc_inv_first=round(auc_inv(hhs,ini),3),auc_inv_min=round(auc_inv(hhs,best),3),
  mean_first=round(ini.mean(),3),mean_min=round(best.mean(),3),share_changed=round(float(np.mean(ini>best+1e-12)),3),max_ha_first=round(ini.max(),3))
# distinct texts
if any(not r["original_text"] for r in src.values()): sys.exit("Redacted texts found: full source texts are required.")
txt={d:src[d]["original_text"] for d in src}; dom={d:src[d]["domain"] for d in src}
tid={}; 
for d,t in txt.items(): tid.setdefault((dom[d],t),len(tid))
doc2t={d:tid[(dom[d],txt[d])] for d in txt}
out["distinct_texts"]=len(tid); out["distinct_by_domain"]=dict(Counter(k[0] for k in tid))
# KL machinery with eps
def kl(a,b,eps=1e-10,n=2):
    pt,qt=_tokens(a,n),_tokens(b,n); v=sorted(set(pt)|set(qt))
    if not v: return 0.0
    cp,cq=Counter(pt),Counter(qt); p=np.array([cp[x] for x in v],float); q=np.array([cq[x] for x in v],float)
    p/=p.sum(); q/=q.sum(); q=(q+eps)/(1+eps*len(v)); m=p>0
    return float(np.sum(p[m]*np.log2(p[m]/q[m])))
# (c) dedup HA: one obs per distinct text (mean of selected scores), HH reconstructed
byt=defaultdict(list)
for r in ha: byt[doc2t[r["document_id"]]].append(float(r["kldi"]))
ha_dd=np.array([np.mean(v) for v in byt.values()]); 
tdom={t:k[0] for k,t in tid.items()}; ttext={t:k[1] for k,t in tid.items()}
rng=np.random.default_rng(2026)
def rebuild(k_per=50,seed=0,eps=1e-10):
    r=np.random.default_rng(seed); vals=[]; 
    for d in sorted(set(tdom.values())):
        ts=[t for t in tdom if tdom[t]==d]; pairs=[(a,b) for a in ts for b in ts if a<b]
        idx=r.choice(len(pairs),min(k_per,len(pairs)),replace=False)
        for i in idx:
            a,b=pairs[i]; vals.append(kl(ttext[a],ttext[b],eps))
    return np.array(vals)
res=[]
for s in range(10):
    v=rebuild(50,s); res.append((tuk(v)[3],auc_inv(v,hk),auc_inv(v,ha_dd),float(v.min()),float(v.max())))
res=np.array(res)
out["HH_rebuilt_random_distinct_10seeds"]=dict(T_mean=round(res[:,0].mean(),2),T_range=[round(res[:,0].min(),2),round(res[:,0].max(),2)],
  AUC_inv_all500_mean=round(res[:,1].mean(),3),AUC_inv_all500_range=[round(res[:,1].min(),3),round(res[:,1].max(),3)],
  AUC_inv_dedupHA_mean=round(res[:,2].mean(),3),AUC_inv_dedupHA_range=[round(res[:,2].min(),3),round(res[:,2].max(),3)],
  HHmin=round(res[:,3].min(),2),HHmax=round(res[:,4].max(),2))
# HH excluding identical pairs (existing sample)
hh_ne=np.array([float(r["kldi"]) for r in hh if r["identical_text"]=="0"])
out["HH_excl_identical"]=dict(n=len(hh_ne),**fmt({k:v for k,v in summ(hh_ne).items() if k!="n"}),AUC_inv=round(auc_inv(hh_ne,hk),3),AUC_inv_dedupHA=round(auc_inv(hh_ne,ha_dd),3),
   above_T=int((hk>tuk(hh_ne)[3]).sum()))
# (d) per domain & equal weight
pd={}
for d in sorted(set(dom.values())):
    a=np.array([float(r["kldi"]) for r in ha if r["domain"]==d]); b=np.array([float(r["kldi"]) for r in hh if r["domain"]==d])
    dd=np.array([np.mean(v) for t,v in byt.items() if tdom[t]==d])
    pd[d]=dict(n_HA=len(a),n_distinct=len(dd),n_HH=len(b),HA_med=round(np.median(a),2),HH_med=round(np.median(b),2),AUC_inv=round(auc_inv(b,a),3),AUC_inv_dedup=round(auc_inv(b,dd),3))
out["per_domain"]=pd
# equal-weight AUC: mean of domain AUCs
out["equal_weight_AUC_inv"]=round(float(np.mean([pd[d]["AUC_inv"] for d in pd])),3)
# (e) cluster bootstrap over distinct texts within domain; HH = all ordered distinct-text pairs among sampled texts
doms=sorted(pd); texts={d:[t for t in tdom if tdom[t]==d] for d in doms}
M={}
for d in doms:
    ts=texts[d]; M[d]=np.array([[kl(ttext[a],ttext[b]) if a!=b else np.nan for b in ts] for a in ts])
ha_by={d:{t:np.array(byt[t]) for t in texts[d]} for d in doms}
def aucinv_pairs(h,a):
    # exact P(a<h)+0.5P(a==h)
    h=np.asarray(h); a=np.asarray(a)
    return float((np.sum(a[:,None]<h[None,:])+0.5*np.sum(a[:,None]==h[None,:]))/(len(a)*len(h)))
# point estimate with full distinct-pair reference
allhh=np.concatenate([M[d][~np.isnan(M[d])] for d in doms]); 
out["HH_all_distinct_ordered_pairs"]=dict(n=len(allhh),**fmt({k:v for k,v in summ(allhh).items() if k!="n"}))
out["AUC_inv_cluster_point"]=round(aucinv_pairs(allhh,np.concatenate([ha_by[d][t] for d in doms for t in texts[d]])),3)
B=1000; boots=[]
for b in range(B):
    hs=[];as_=[]
    for d in doms:
        n=len(texts[d]); idx=rng.integers(0,n,n)
        sub=M[d][np.ix_(idx,idx)]; same=(np.array(idx)[:,None]==np.array(idx)[None,:])
        hs.append(sub[~same & ~np.isnan(sub)]); as_+= [ha_by[d][texts[d][i]] for i in idx]
    boots.append(aucinv_pairs(np.concatenate(hs),np.concatenate(as_)))
out["AUC_inv_cluster_boot_95CI"]=[round(float(np.percentile(boots,2.5)),3),round(float(np.percentile(boots,97.5)),3)]
# cluster-level permutation-free MWU analog: report effect via cluster bootstrap only
# (f) smoothing + JSD/TV + length diagnostics
def dists(a,b,n=2):
    pt,qt=_tokens(a,n),_tokens(b,n); v=sorted(set(pt)|set(qt)); cp,cq=Counter(pt),Counter(qt)
    p=np.array([cp[x] for x in v],float); q=np.array([cq[x] for x in v],float); return p/p.sum(),q/q.sum(),pt,qt
def jsd(p,q):
    m=(p+q)/2; f=lambda x,y:np.sum(x[x>0]*np.log2(x[x>0]/y[x>0])); return 0.5*f(p,m)+0.5*f(q,m)
rows=defaultdict(list)
# HH pairs actually used
hhpairs=[(src[r["document_id_a"]]["original_text"],src[r["document_id_b"]]["original_text"],r["identical_text"]) for r in hh]
hapairs=[(src[r["document_id"]]["original_text"],par[r["document_id"]]) for r in ha]
def run(pairs):
    o=defaultdict(list)
    for a,b in pairs[:]:
        for eps in (1e-10,1e-6,1e-3):
            o[f"kl_eps{eps:g}"].append(kl(a,b,eps))
        p,q,pt,qt=dists(a,b); o["jsd"].append(jsd(p,q)); o["tv"].append(0.5*np.abs(p-q).sum())
        sa,sb=set(pt),set(qt); o["bigram_overlap"].append(len(sa&sb)/len(sa|sb) if sa|sb else 1.0)
        o["src_bigrams"].append(len(pt)); o["src_words"].append(len(a.split())); 
    return {k:np.array(v) for k,v in o.items()}
HAm=run(hapairs); HHm=run([(a,b) for a,b,i in hhpairs if i=="0"]); HHall=run([(a,b) for a,b,i in hhpairs])
sens={}
for k in ["kl_eps1e-10","kl_eps1e-06","kl_eps0.001","jsd","tv","bigram_overlap"]:
    pos_low = k!="bigram_overlap"
    # inverted AUC for distance measures: P(HA<HH); for overlap: P(HA>HH)
    h,a=HHm[k],HAm[k]
    A=aucinv_pairs(h,a) if pos_low else aucinv_pairs(a,h)
    sens[k]=dict(HA_med=round(float(np.median(a)),3),HH_med=round(float(np.median(h)),3),AUC_separating=round(A,3),T_tukey=round(tuk(h)[3],2),HA_above_T=int((a>tuk(h)[3]).sum()) if pos_low else None,HH_max=round(float(h.max()),2),HA_max=round(float(a.max()),2))
out["sensitivity_nonidentical_HH"]=sens
out["length"]=dict(HA_src_words_med=float(np.median(HAm["src_words"])),HA_src_words_min_max=[int(HAm["src_words"].min()),int(HAm["src_words"].max())],
  HA_src_bigrams_med=float(np.median(HAm["src_bigrams"])),HA_src_bigrams_min_max=[int(HAm["src_bigrams"].min()),int(HAm["src_bigrams"].max())],
  HA_bigram_overlap_med=round(float(np.median(HAm["bigram_overlap"])),3),HH_bigram_overlap_med=round(float(np.median(HHm["bigram_overlap"])),3),
  HH_share_zero_overlap=round(float(np.mean(HHm["bigram_overlap"]==0)),3))
out["ceiling_bits"]=round(float(np.log2((1+1e-10*60)/1e-10)),2)
# lang/prose check: german passages in law
de=sum(1 for d in txt if dom[d]=="law" and any(w in txt[d].lower() for w in (" der "," die "," und "," das "," nicht ")))
out["law_passages_with_german_markers"]=de
out["urls"]={u:c for u,c in Counter((src[d]["source_url"].split('/')[2]+"..."+src[d]["source_url"][-25:]) for d in src).items()}

out["AUC_inv_dedupHA_origHH"]=round(auc_inv(hhs,ha_dd),3)
print(json.dumps(out,indent=1,default=float))
