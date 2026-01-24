# GPU Signal Tracker  
*A signal-first approach to understanding NVIDIA & GPU market momentum*

---

## Why I built this

GPU and AI news moves fast — too fast to read everything and still know what actually matters.

I kept running into the same problem:
- Lots of headlines  
- Lots of noise  
- Very little signal  

This project started as a way to answer a simple question:

> **“What themes are actually shaping the GPU market right now?”**

Instead of reading articles one by one, I wanted a system that:
- pulls fresh headlines automatically  
- turns unstructured news into structured data  
- highlights momentum instead of anecdotes  
- stays interpretable (not a black box)

This repo is the result.

---

## What this project does

The GPU Signal Tracker is a small pipeline that:

1. **Fetches GPU-related news** from trusted sources (NVIDIA, Reuters, The Verge, Tom’s Hardware)
2. **Tags each headline** into business-relevant themes
3. **Uses AI to refine ambiguous cases** instead of forcing brittle rules
4. **Weights and filters results by time**
5. **Outputs visual trends and written summaries** (daily or weekly)

The goal is not prediction — it’s **decision clarity**.

---

## Core ideas behind the design

### 1. Rules first, AI second  
Keyword rules are fast, transparent, and reliable for obvious cases.  
AI is only used where rules break down (the “other” bucket).

This hybrid approach keeps the system:
- explainable  
- debuggable  
- extensible  

### 2. Momentum > volume  
A topic mentioned twice this week can matter more than one mentioned ten times last quarter.

All analysis is:
- time-aware (last 7 days / daily)  
- focused on **recent change**, not historical noise  

### 3. Not all sources are equal  
First-party announcements and authoritative reporting matter more than general media.

The pipeline applies **source weighting** so important signals carry more influence.

---

## Tag taxonomy

The system classifies headlines into these themes:

- **cloud_partnerships**  
  AWS, Azure, GCP, platform integrations, strategic partnerships

- **ai_demand**  
  Demand signals, backlog, capacity pressure, adoption trends

- **infrastructure_expansion**  
  Datacenter buildouts, capex, factories, supply chain expansion

- **product_launch**  
  GPU / chip / platform announcements and releases

- **earnings**  
  Earnings, guidance, revenue, margins, quarterly results

- **competition**  
  AMD, Intel, alternative accelerators, competitive positioning

- **regulation_export**  
  Export controls, sanctions, regulation, China policy

- **other**  
  Anything that doesn’t confidently fit the above

---

## How the pipeline works

### 1. Fetch headlines  
`fetch_rss.py` pulls the latest headlines from multiple RSS feeds and writes them to a structured CSV.

### 2. Initial rule-based tagging  
Explicit keywords handle high-confidence classifications.

### 3. AI-assisted reclassification  
`ai_tag.py`:
- looks only at headlines tagged as `other`
- asks an LLM to assign the **best existing label**
- applies a confidence threshold before overriding

This avoids overusing AI while still improving coverage.

### 4. Visualization  
`plot.py` creates charts showing:
- weighted signal momentum  
- recent activity only  
- optional confidence segmentation  

### 5. Summaries (optional)  
`summarize_brief.py` generates a **daily or weekly GPU briefing** that reads like a short market memo.

---

## Example outputs

- `data/signals.csv` – raw, structured headlines  
- `data/signals_ai.csv` – AI-refined dataset  
- `data/tag_counts.png` – weighted momentum chart  
- `data/briefing_weekly.md` – written weekly summary  

---

## How to run it

```bash
pip install -r requirements.txt

python src/fetch_rss.py
python src/ai_tag.py
python src/plot.py
python src/summarize_brief.py weekly
