import re,sys
def parse(f):
    txt=open(f,errors='ignore').read().replace('\r','\n')
    lc=None; rows={}
    for line in txt.split('\n'):
        if '[LONGCTX]' in line:
            lc={k:float(v) for k,v in re.findall(r'(never|<128|128-511|512-2k|>2k)=([\d.]+)\(',line)}
            lc['n']={k:int(v) for k,v in re.findall(r'(never|<128|128-511|512-2k|>2k)=[\d.]+\(n(\d+)\)',line)}
        m=re.search(r'epoch (\d+)\s+\(.*val_ppl=([\d.]+)',line)
        if m: rows[int(m.group(1))]=(float(m.group(2)),lc); lc=None
    return rows
a=parse('logs/text_glob400_levelsep_384.log'); b=parse('logs/text_transformer_wikitext_d384_4k.log')
K=['never','<128','128-511','512-2k','>2k']
print('ep | ppl P / T | '+' | '.join(f'{k} P/T' for k in K))
for e in sorted(set(a)&set(b)):
    if e%5 and e<max(set(a)&set(b))-4: continue
    pa,la=a[e]; pb,lb=b[e]
    s=' | '.join(f'{la[k]:.1f}/{lb[k]:.1f}' if la and lb else '-' for k in K)
    print(f'{e:3d} | {pa:.2f} / {pb:.2f} | {s}')
print('last', max(a), max(b), 'best P', min(v[0] for v in a.values()), 'best T', min(v[0] for v in b.values()))
# ratio averaged over last 10 common epochs (log-ratio mean)
import math
com=sorted(set(a)&set(b))[-10:]
for k in K:
    r=[math.log(a[e][1][k]/b[e][1][k]) for e in com if a[e][1] and b[e][1]]
    print(k, 'P/T geo-mean ratio last10:', round(math.exp(sum(r)/len(r)),3), 'n', a[com[-1]][1]['n'][k])
r=[math.log(a[e][0]/b[e][0]) for e in com]; print('overall', round(math.exp(sum(r)/len(r)),3))
# same-seq check: n counts equal?
print('n equal across models at last epoch:', a[com[-1]][1]['n']==b[com[-1]][1]['n'], a[com[-1]][1]['n'], b[com[-1]][1]['n'])
