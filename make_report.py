#!/usr/bin/env python3
"""把 sweep 结果转成一个自包含的 HTML 报告。

功能：
  1. 枢轴原文用 <pre> 原样显示（不被 markdown 打断）；
  2. 附英文题面 + 英文参考 + 日语定理词高亮；
  3. 【英文机翻】把每条枢轴选段翻成英文（用 Qwen3，带缓存）；
  4. 【标记 + 筛选】点击标记某组合，可只看已标记（localStorage 持久化）。

用法：
    python make_report.py                 # 翻译 + 生成（首次约几分钟，缓存后快）
    python make_report.py --skip-translate  # 跳过翻译，快速生成
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
SWEEP_DIR = BASE / "outputs" / "sweep_hyperparams"
DATA_PATH = BASE / "data" / "seed_ja.jsonl"
CACHE_PATH = SWEEP_DIR / "translation_cache.json"

THEOREMS = {
    0: ("Vieta's formulas", ["解と係数", "ビエタ", "ヴィエタ"]),
    1: ("Pythagorean theorem", ["ピタゴラス", "三平方"]),
    2: ("law of cosines", ["余弦定理", "コサイン"]),
    3: ("pigeonhole principle", ["鳩の巣"]),
    4: ("binomial theorem", ["二項定理"]),
    5: ("factor theorem", ["因数定理"]),
}


def parse_summary(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "window_k" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        wk, layer, vd = int(cells[0]), cells[1], int(cells[2])
        rows[(wk, layer, vd)] = {"window_k": wk, "layer": layer, "vec_dim": vd,
                                 "avg_contrast": cells[3], "avg_prom": cells[4],
                                 "avg_kurt": cells[5], "avg_n_peaks": cells[6]}
    return rows


def parse_pivots(path: Path):
    combos, cur, cur_example = [], None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            m = re.search(r"k=(\d+), layer=(\[[\d.,]+\)|\[[\d.,]+\)), vec_dim=(\d+)", line)
            if m:
                cur = [(int(m.group(1)), m.group(2), int(m.group(3))), []]
                combos.append(cur)
            i += 1
        elif line.startswith("### "):
            m = re.match(r"### 例 (\d+)", line)
            if m and cur is not None:
                cur_example = [int(m.group(1)), []]
                cur[1].append(cur_example)
            i += 1
        elif line.startswith("- t="):
            if cur is not None and cur_example is not None:
                pidx, zv, ctx, i = _parse_spike(lines, i)
                cur_example[1].append((pidx, zv, ctx))
            else:
                i += 1
        else:
            i += 1
    return combos


def _parse_spike(lines, i):
    head, _, rest = lines[i].partition("…")
    ctx_parts = [rest] if "…" in lines[i] else []
    i += 1
    while i < len(lines):
        line = lines[i]
        if "…" in line:
            ctx_parts.append(line.partition("…")[0])
            i += 1
            break
        if line.startswith("- t=") or line.startswith("### ") or line.startswith("## "):
            break
        ctx_parts.append(line)
        i += 1
    ctx = "\n".join(ctx_parts).strip()
    m = re.match(r"- t=(\d+) z=([\d.]+)", head)
    return int(m.group(1)), float(m.group(2)), ctx, i


def highlight(text: str, keywords: list[str]) -> str:
    out = html.escape(text)
    for kw in keywords:
        out = out.replace(kw, f"<mark>{kw}</mark>")
    return out


def collect_unique_contexts(combos):
    seen = set()
    out = []
    for _key, examples in combos:
        for _ex, spikes in examples:
            for _p, _z, ctx in spikes:
                if ctx and ctx not in seen:
                    seen.add(ctx)
                    out.append(ctx)
    return out


def translate_texts(model, tokenizer, texts, cache, max_new_tokens=96):
    """用 Qwen3 把日语片段翻成英文，带缓存。返回 {text: english}。"""
    todo = [t for t in texts if t not in cache]
    if todo:
        print(f"  翻译 {len(todo)} 条唯一枢轴片段 ...")
        for k, t in enumerate(todo, 1):
            try:
                en = _translate_one(model, tokenizer, t, max_new_tokens)
                cache[t] = en
            except Exception:
                cache[t] = ""
            if k % 50 == 0:
                print(f"    {k}/{len(todo)}")
                json.dump(cache, CACHE_PATH.open("w", encoding="utf-8"), ensure_ascii=False)
        json.dump(cache, CACHE_PATH.open("w", encoding="utf-8"), ensure_ascii=False)
    return cache


def _translate_one(model, tokenizer, text, max_new_tokens):
    import torch
    device = next(model.parameters()).device
    prompt = ("Translate the Japanese to English, keeping math symbols (x, ^2, \\boxed, etc.) unchanged. "
              "Output only the English translation, no extra text.\n\nJapanese:\n" + text + "\n\nEnglish:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gcfg = model.generation_config
    gcfg.do_sample = False
    gcfg.temperature = None
    gcfg.top_p = None
    gcfg.top_k = None
    gcfg.max_new_tokens = max_new_tokens
    gcfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = model.generate(**inputs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build_html(summary, combos, seeds, translations) -> str:
    esc = html.escape
    summary_sorted = sorted(summary.items(), key=lambda kv: -float(kv[1]["avg_contrast"]))
    p = []
    p.append("<!doctype html><html><head><meta charset='utf-8'>")
    p.append("<style>"
             "body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:1200px;margin:1.5rem auto;padding:0 1rem;color:#1a1a1a;}"
             ".controls{margin:.8rem 0;display:flex;gap:1rem;flex-wrap:wrap;align-items:center}"
             "table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.6rem 0}"
             "th,td{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}"
             "th{background:#f5f5f5} tr.clickable{cursor:pointer} tr.clickable:hover{background:#f0f7ff}"
             "section.combo{border-top:2px solid #ccc;padding:1rem 0;margin-top:.5rem}"
             "section.combo.marked{background:#f0fff0;border-left:5px solid #2ecc71;padding-left:.6rem}"
             "h2{font-size:1.05rem;margin:.3rem 0;display:flex;align-items:center;gap:.6rem}"
             ".markbtn{cursor:pointer;font-size:.8rem;padding:.1rem .5rem;border:1px solid #2ecc71;border-radius:4px;background:#fff}"
             ".markbtn.on{background:#2ecc71;color:#fff}"
             ".example{margin:.8rem 0;padding:.6rem;background:#fafafa;border-radius:6px}"
             "h3{font-size:.95rem;margin:.3rem 0} .en{font-size:.85rem;color:#333;margin:.2rem 0} .en b{color:#8a2be2}"
             ".spike{display:flex;gap:.6rem;margin:.4rem 0;align-items:flex-start}"
             ".meta{font-size:.8rem;color:#c0392b;white-space:nowrap;font-weight:600;min-width:100px}"
             "pre{white-space:pre-wrap;word-break:break-word;margin:0;font-family:ui-monospace,Menlo,monospace;font-size:.82rem;background:#fff;border:1px solid #eee;padding:.4rem .5rem;border-radius:4px;flex:1}"
             "pre.en{background:#f0f7ff;border-color:#cfe3ff;color:#0a3d62}"
             "mark{background:#ffe58a;padding:0 2px;border-radius:3px}"
             ".hidden{display:none}"
             "</style></head><body>")
    p.append("<h1>PDSD 超参数扫描报告</h1>")
    p.append("<div class='controls'>"
             "<label>vec_dim：<select id='f-vec'><option value=''>全部</option></select></label>"
             "<label>window_k：<select id='f-k'><option value=''>全部</option></select></label>"
             "<label>layer：<select id='f-layer'><option value=''>全部</option></select></label>"
             "<label><input type='checkbox' id='f-mark'> 只看含定理关键词</label>"
             "<label><input type='checkbox' id='f-fav'> 只看已标记</label>"
             "<button onclick='exportMarks()'>导出标记</button>"
             "</div>")

    p.append("<table><tr><th>排名</th><th>window_k</th><th>layer</th><th>vec_dim</th>"
             "<th>avg_contrast</th><th>avg_prom</th><th>avg_kurt</th><th>avg_n_peaks</th></tr>")
    for rank, (key, m) in enumerate(summary_sorted, 1):
        wk, layer, vd = key
        p.append(f"<tr class='clickable' data-key='{wk}|{layer}|{vd}' onclick=\"jump('{wk}|{layer}|{vd}')\">"
                 f"<td>{rank}</td><td>{wk}</td><td>{esc(layer)}</td><td>{vd}</td>"
                 f"<td>{esc(m['avg_contrast'])}</td><td>{esc(m['avg_prom'])}</td>"
                 f"<td>{esc(m['avg_kurt'])}</td><td>{esc(m['avg_n_peaks'])}</td></tr>")
    p.append("</table>")

    for key, examples in combos:
        wk, layer, vd = key
        m = summary.get(key, {})
        p.append(f"<section class='combo' id='{wk}|{layer}|{vd}' data-k='{wk}' data-vec='{vd}' data-layer='{esc(layer)}'>")
        p.append(f"<h2>k={wk} layer={layer} vec_dim={vd} "
                 f"(contrast={esc(m.get('avg_contrast','?'))}, n_peaks={esc(m.get('avg_n_peaks','?'))}) "
                 f"<button class='markbtn' onclick='toggleMark(this)'>标记</button></h2>")
        for ex_idx, spikes in examples:
            seed = seeds[ex_idx]
            en_name, ja_kws = THEOREMS.get(ex_idx, ("", []))
            p.append("<div class='example'>")
            p.append(f"<h3>例 {ex_idx}：{esc(seed['problem_ja'][:50])}</h3>")
            p.append(f"<p class='en'><b>定理：{esc(en_name)}</b></p>")
            p.append(f"<p class='en'>题（英）：{esc(seed['problem'])}</p>")
            p.append(f"<p class='en'>参考（英）：{esc(seed['solution'])}</p>")
            for pidx, zv, ctx in spikes:
                en = translations.get(ctx, "")
                p.append(f"<div class='spike'><span class='meta'>t={pidx} z={zv:.2f}</span>"
                         f"<pre>{highlight(ctx, ja_kws)}</pre></div>")
                if en:
                    p.append(f"<div class='spike' style='margin-top:-.3rem'><span class='meta' style='color:#888'>EN</span>"
                             f"<pre class='en'>{esc(en)}</pre></div>")
            p.append("</div>")
        p.append("</section>")

    p.append("<script>"
             "const secs=[...document.querySelectorAll('section.combo')];"
             "const vecs=[...new Set(secs.map(s=>s.dataset.vec))].sort((a,b)=>a-b);"
             "const ks=[...new Set(secs.map(s=>s.dataset.k))].sort((a,b)=>a-b);"
             "const layers=[...new Set(secs.map(s=>s.dataset.layer))];"
             "function fill(sel,vals){vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o)});}"
             "fill(document.getElementById('f-vec'),vecs);fill(document.getElementById('f-k'),ks);fill(document.getElementById('f-layer'),layers);"
             "const marks=new Set(JSON.parse(localStorage.getItem('marks')||'[]'));"
             "function toggleMark(btn){const sec=btn.closest('section.combo');const key=sec.id;if(marks.has(key)){marks.delete(key);sec.classList.remove('marked');btn.classList.remove('on');btn.textContent='标记';}else{marks.add(key);sec.classList.add('marked');btn.classList.add('on');btn.textContent='已标记';}localStorage.setItem('marks',JSON.stringify([...marks]));apply();}"
             "secs.forEach(s=>{if(marks.has(s.id)){s.classList.add('marked');const b=s.querySelector('.markbtn');b.classList.add('on');b.textContent='已标记';}});"
             "function apply(){"
             "const fv=document.getElementById('f-vec').value,fk=document.getElementById('f-k').value,fl=document.getElementById('f-layer').value,fm=document.getElementById('f-mark').checked,ff=document.getElementById('f-fav').checked;"
             "secs.forEach(s=>{const ok=(!fv||s.dataset.vec===fv)&&(!fk||s.dataset.k===fk)&&(!fl||s.dataset.layer===fl)&&(!ff||marks.has(s.id));s.classList.toggle('hidden',!ok);});"
             "if(fm){document.querySelectorAll('.spike').forEach(sp=>sp.classList.toggle('hidden',!sp.querySelector('mark')));}else{document.querySelectorAll('.spike').forEach(sp=>sp.classList.remove('hidden'));}"
             "}"
             "['f-vec','f-k','f-layer','f-mark','f-fav'].forEach(id=>document.getElementById(id).onchange=apply);"
             "function jump(key){const el=document.getElementById(key);if(el){el.scrollIntoView({behavior:'smooth'});el.style.background='#fffbe6';setTimeout(()=>el.style.background='',1500);}}"
             "function exportMarks(){const list=[...marks].sort().map(k=>{const p=k.split('|');return 'k='+p[0]+', layer='+p[1]+', vec_dim='+p[2];}).join('\\n');if(!list){alert('还没有标记任何组合');return;}prompt('复制下面的已标记组合（Ctrl+A 全选 → Ctrl+C）:',list);}"
             "</script></body></html>")
    return "".join(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-translate", action="store_true", help="跳过机翻，快速生成")
    ap.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = parse_summary(SWEEP_DIR / "summary.md")
    combos = parse_pivots(SWEEP_DIR / "pivots.md")

    translations = {}
    if CACHE_PATH.exists():
        translations = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if not args.skip_translate:
        import torch
        from seed_builder import load_model
        print(f"加载模型 {args.model_name} 用于翻译 ...")
        model, tokenizer = load_model(args.model_name, torch.bfloat16)
        contexts = collect_unique_contexts(combos)
        translations = translate_texts(model, tokenizer, contexts, translations)

    html_str = build_html(summary, combos, seeds, translations)
    out = SWEEP_DIR / "report.html"
    out.write_text(html_str, encoding="utf-8")
    n_tr = sum(1 for v in translations.values() if v)
    print(f"已生成 {out}（{len(combos)} 组合，翻译 {n_tr} 条片段）。浏览器打开即可。")


if __name__ == "__main__":
    main()
