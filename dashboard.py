"""DualNet-AL results dashboard:  streamlit run dashboard.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config('DualNet-AL', layout='wide')
st.title('DualNet-AL: active-learning results')

runs = sorted((Path(__file__).parent / 'results').glob('*.json'),
              key=lambda p: (p.stem.startswith('smoke_'), -p.stat().st_mtime))  # newest real run first
if not runs:
    st.info('No results yet. Run `python dualnet_al.py` first.')
    st.stop()
run = st.sidebar.selectbox('Run', runs, format_func=lambda p: p.stem)
res = json.loads(run.read_text())
st.sidebar.json(res['config'], expanded=False)

# fixed hue per strategy (validated light/dark steps); DualNet-AL takes categorical slot 1
PALETTE = {'light': dict(dualnet='#2a78d6', entropy='#eb6834', coreset='#1baf7a', random='#eda100'),
           'dark': dict(dualnet='#3987e5', entropy='#d95926', coreset='#199e70', random='#c98500')}
colors = PALETTE['dark' if st.context.theme.type == 'dark' else 'light']
hues = lambda df: [colors[s] for s in df.columns]

rows = [dict(strategy=s, seed=k, **{m: h[m] for m in ('round', 'n', 'acc', 'macro_f1', 'tail_f1', 'imbalance')})
        for s, seeds in res['results'].items() for k, hist in enumerate(seeds) for h in hist]
mean = pd.DataFrame(rows).groupby(['strategy', 'n']).mean(numeric_only=True).reset_index()

for col, (m, label) in zip(st.columns(3), [('acc', 'Test accuracy vs labels'),
                                           ('tail_f1', 'Tail-class F1 vs labels'),
                                           ('imbalance', 'Labelled-pool imbalance (max/min)')]):
    col.subheader(label)
    curves = mean.pivot(index='n', columns='strategy', values=m)
    col.line_chart(curves, color=hues(curves))

st.subheader('Summary (mean ± std over seeds)')
summary = pd.DataFrame({s: {m: f'{mu:.3f} ± {sd:.3f}' for m, (mu, sd) in d.items()}
                        for s, d in res['summary'].items()}).T
st.dataframe(summary, width='stretch')

st.subheader('Per-class view')
r = st.slider('Round', 0, res['config']['rounds'], res['config']['rounds'])
per_class = lambda key: pd.DataFrame({s: np.mean([h[r][key] for h in seeds], 0)
                                      for s, seeds in res['results'].items()})
c1, c2 = st.columns(2)
c1.caption('Labels per class in the labelled pool (class 0 = most frequent in the pool)')
counts, f1 = per_class('counts'), per_class('f1')
c1.bar_chart(counts, stack=False, color=hues(counts))
c2.caption('Per-class F1 on the balanced test set')
c2.bar_chart(f1, stack=False, color=hues(f1))

st.download_button('Download summary CSV', summary.to_csv().encode(), f'{run.stem}_summary.csv')
st.download_button('Download raw results JSON', run.read_bytes(), run.name)
